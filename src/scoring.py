"""
Module d'analyse sémantique pour l'orientation professionnelle.
Utilise SentenceTransformers (bi-encoder) + CrossEncoder pour le matching compétences/métiers.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer, util, CrossEncoder


# ============================================================================
# MODÈLES & DONNÉES (CACHE)
# ============================================================================

@st.cache_resource
def get_models() -> Tuple[SentenceTransformer, CrossEncoder]:
    """
    Charge les modèles de NLP en cache pour éviter de recharger à chaque requête.

    Returns:
        bi_model: Modèle bi-encodeur pour embedding sémantique
        cross_model: Cross-encoder pour re-ranking précis
    """
    bi_model = SentenceTransformer('all-MiniLM-L6-v2')
    cross_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
    return bi_model, cross_model


@st.cache_data
def load_and_index_data(json_path: str) -> Tuple[Dict, Dict, List[str], np.ndarray]:
    """
    Charge le référentiel métier et pré-calcule les embeddings des compétences.

    Args:
        json_path: Chemin vers le fichier JSON du référentiel

    Returns:
        data: Référentiel complet (blocs, compétences, métiers)
        comp_index: Index {comp_id: compétence}
        comp_ids: Liste ordonnée des IDs de compétences
        embeddings: Matrice d'embeddings des compétences
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    comp_index = {}
    for bloc in data['blocs']:
        for comp in bloc['competences']:
            cid = comp['id']
            comp_index[cid] = {
                'texte': comp['texte'],
                'bloc_id': bloc['id'],
                'bloc_nom': bloc['nom']
            }

    bi_model, _ = get_models()
    comp_ids = list(comp_index.keys())
    texts = [comp_index[cid]['texte'] for cid in comp_ids]

    embeddings = bi_model.encode(
        texts,
        convert_to_tensor=True,
        show_progress_bar=False,
        batch_size=64
    )

    return data, comp_index, comp_ids, embeddings


# ============================================================================
# ANALYSE SÉMANTIQUE
# ============================================================================

def sigmoid(x: float) -> float:
    """Normalise un score brut en probabilité [0, 1]."""
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))


def analyze_profile(
        user_text: str,
        bi_model: SentenceTransformer,
        comp_ids: List[str],
        comp_embeddings: np.ndarray,
        cross_model: CrossEncoder,
        comp_index: Dict,
        top_k: int = 40
) -> Dict[str, float]:
    """
    Analyse le profil utilisateur et calcule les scores de match par compétence.

    Pipeline RAG:
    1. Bi-encoder : recherche sémantique rapide (top_k candidats)
    2. Cross-encoder : re-ranking précis des candidats
    3. Normalisation min-max : scores entre 0 et 1

    Args:
        user_text: Description du profil utilisateur (concaténée)
        bi_model: Modèle d'embedding
        comp_ids: Liste des IDs de compétences
        comp_embeddings: Embeddings pré-calculés des compétences
        cross_model: Modèle de re-ranking
        comp_index: Index des compétences
        top_k: Nombre de compétences candidates à re-ranker

    Returns:
        Dictionnaire {comp_id: score_normalisé}
    """
    # Étape 1 : Embedding du profil utilisateur
    user_emb = bi_model.encode(user_text, convert_to_tensor=True)

    # Étape 2 : Recherche sémantique avec bi-encoder
    hits = util.semantic_search(user_emb, comp_embeddings, top_k=top_k)

    candidate_indices = [hit['corpus_id'] for hit in hits[0]]
    candidate_ids = [comp_ids[idx] for idx in candidate_indices]
    candidate_texts = [comp_index[cid]['texte'] for cid in candidate_ids]

    # Étape 3 : Re-ranking avec Cross-Encoder
    pairs = [[user_text, text] for text in candidate_texts]
    cross_scores = cross_model.predict(pairs)

    # ✅ CORRECTION : Normalisation Min-Max au lieu de Sigmoid
    # Les scores bruts du cross-encoder sont souvent entre -10 et +10
    # On normalise entre 0 et 1 de manière linéaire

    min_score = float(np.min(cross_scores))
    max_score = float(np.max(cross_scores))

    print(f"\n🔍 DEBUG analyze_profile:")
    print(f"  - Cross-encoder scores: min={min_score:.3f}, max={max_score:.3f}")

    # Si tous les scores sont identiques (rare), on attribue 0.5 partout
    if max_score == min_score:
        normalized_scores = [0.5] * len(cross_scores)
    else:
        # Normalisation Min-Max : (x - min) / (max - min)
        normalized_scores = [
            (score - min_score) / (max_score - min_score)
            for score in cross_scores
        ]

    # ✅ BOOST : Appliquer une transformation pour augmenter les scores moyens
    # Les scores normalisés sont souvent trop bas, on les "boost" avec une puissance
    BOOST_FACTOR = 0.7  # Plus c'est bas (0.5-0.8), plus les scores sont boostés
    boosted_scores = [score ** BOOST_FACTOR for score in normalized_scores]

    result = {
        candidate_ids[idx]: float(boosted_scores[idx])
        for idx in range(len(candidate_ids))
    }

    # DEBUG : Afficher les top résultats
    top_results = sorted(result.items(), key=lambda x: -x[1])[:5]
    print(f"  - Top 5 compétences finales :")
    for cid, score in top_results:
        print(f"    * {comp_index[cid]['texte']}: {score:.2%}")

    return result


# ============================================================================
# RECOMMANDATION MÉTIERS
# ============================================================================

def recommend_jobs(
    comp_scores: Dict[str, float],
    data: Dict,
    top_n: int = 3
) -> Tuple[List[Dict], Dict[str, float]]:
    """
    Recommande les métiers les plus adaptés au profil utilisateur.

    Agrégation: moyenne des scores des compétences par bloc,
    puis moyenne pondérée des blocs requis par métier.

    Args:
        comp_scores: Scores de match par compétence
        data: Référentiel métier complet
        top_n: Nombre de métiers à recommander

    Returns:
        top_jobs: Liste des top_n métiers {titre, description, score, blocs}
        bloc_scores: Scores moyens par bloc (pour visualisation)
    """
    comp_to_bloc = {}
    bloc_index = {}

    for bloc in data['blocs']:
        bloc_id = bloc['id']
        bloc_index[bloc_id] = bloc
        for comp in bloc['competences']:
            comp_id = comp['id']
            comp_to_bloc[comp_id] = bloc_id

    bloc_scores_raw = {b['id']: [] for b in data['blocs']}
    for cid, score in comp_scores.items():
        if cid in comp_to_bloc:
            bid = comp_to_bloc[cid]
            bloc_scores_raw[bid].append(score)

    bloc_scores = {
        bid: float(np.mean(scores)) if scores else 0.0
        for bid, scores in bloc_scores_raw.items()
    }

    recommendations = []
    for job in data['metiers']:
        bloc_ids = job.get('blocs_requis', [])
        if not bloc_ids:
            continue

        job_bloc_scores = [bloc_scores.get(bid, 0.0) for bid in bloc_ids]
        job_score = float(np.mean(job_bloc_scores))

        job_blocs = [
            {
                'nom': bloc_index[bid]['nom'],
                'score': bloc_scores.get(bid, 0.0)
            }
            for bid in bloc_ids
        ]

        recommendations.append({
            'titre': job['titre'],
            'description': job.get('description', ''),
            'score': job_score,
            'blocs': job_blocs
        })

    top_jobs = sorted(recommendations, key=lambda x: x['score'], reverse=True)[:top_n]

    return top_jobs, bloc_scores
