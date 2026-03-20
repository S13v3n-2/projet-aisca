# Moteur de scoring sémantique — matching compétences / métiers
# Modèle : CamemBERT fine-tuné sur notre référentiel (S13v3n-2/scoring-camembert-v2)
# La pondération des niveaux Likert repose sur une moyenne pondérée de vecteurs
# plutôt que sur de la répétition de texte — plus précis et nettement plus rapide.

import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer, util


@st.cache_resource
def get_models() -> SentenceTransformer:
    # Chargement unique au démarrage — HuggingFace met le modèle en cache local
    # après le premier téléchargement, donc pas de latence aux lancements suivants.
    bi_model = SentenceTransformer('S13v3n-2/scoring-camembert-v5')
    return bi_model


@st.cache_data
def load_and_index_data(json_path: str) -> Tuple[Dict, Dict, List[str], np.ndarray]:
    # Lecture du référentiel et construction d'un index plat id -> compétence.
    # On aplatit la structure blocs/compétences pour faire des lookups O(1)
    # pendant le scoring, sans re-parcourir le JSON à chaque analyse.
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    comp_index = {}
    for bloc in data['blocs']:
        for comp in bloc['competences']:
            cid = comp['id']
            comp_index[cid] = {
                'texte':    comp['texte'],
                'niveau':   comp['niveau'],   # 'technique' | 'transversal' | 'interet'
                'bloc_id':  bloc['id'],
                'bloc_nom': bloc['nom']
            }

    bi_model = get_models()
    comp_ids = list(comp_index.keys())
    texts    = [comp_index[cid]['texte'] for cid in comp_ids]

    # Les embeddings sont précalculés ici une bonne fois pour toutes.
    # L'ordre de comp_ids et celui des lignes dans la matrice sont identiques —
    # c'est ce qui permet de retrouver l'ID d'une compétence à partir de son
    # indice dans les résultats de semantic_search.
    embeddings = bi_model.encode(
        texts,
        convert_to_tensor=True,
        show_progress_bar=False,
        batch_size=64
    )

    return data, comp_index, comp_ids, embeddings


# Poids associés à chaque niveau Likert.
# On n'utilise pas une échelle linéaire (0/1/2/3) parce qu'elle traiterait
# le saut Intermédiaire->Avancé et le saut Avancé->Expert comme équivalents,
# ce qui ne correspond pas à la réalité du terrain. Un Expert a une maîtrise
# qualitativement différente d'un Avancé — d'où la progression 0.25 -> 0.60 -> 1.0.
# Le niveau Débutant est volontairement ignoré : inclure un signal très faible
# introduit plus de bruit qu'il n'apporte d'information utile.
LIKERT_WEIGHTS = {
    "Débutant":      0.0,
    "Intermédiaire": 0.25,
    "Avancé":        0.60,
    "Expert":        1.0,
}

# Phrases descriptives par domaine, rédigées avec le vocabulaire exact du référentiel.
# Chaque phrase sert d'ancre sémantique pour son domaine : elle est encodée une seule
# fois au démarrage et son vecteur est pondéré selon le niveau déclaré par l'utilisateur.
DOMAINE_PHRASES = {
    # Bloc B02 — Business et Stratégie
    "business": (
        "Je maîtrise la stratégie d'entreprise et la définition de business models innovants. "
        "Je réalise des analyses de marché, des études de faisabilité technique et financière. "
        "Je pilote des projets complexes avec méthodes agiles, je conduis des analyses concurrentielles "
        "et j'accompagne le changement et les transformations organisationnelles."
    ),
    # Bloc B01 — Juridique
    "juridique": (
        "Je maîtrise le droit des affaires, le droit commercial et le droit du travail. "
        "J'analyse et interprète des contrats et textes juridiques complexes. "
        "Je rédige des actes juridiques et documents légaux professionnels. "
        "J'assure la conformité réglementaire RGPD et le respect des normes. "
        "J'effectue une veille réglementaire et juridique continue. "
        "Je protège la propriété intellectuelle et gère les procédures de contentieux."
    ),
    # Bloc B04 — Finance et Comptabilité
    "finance": (
        "Je maîtrise la comptabilité générale et analytique en normes françaises et IFRS. "
        "Je réalise des analyses financières et diagnostics économiques approfondis. "
        "J'élabore et suis les budgets prévisionnels et contrôle les écarts. "
        "Je crée des tableaux de bord financiers pour le pilotage décisionnel. "
        "Je gère la trésorerie et évalue la rentabilité des investissements."
    ),
    # Bloc B06 — Création et Design
    "design": (
        "Je maîtrise le design graphique et la création d'identité visuelle complète. "
        "J'assure la direction artistique de projets créatifs complexes. "
        "Je crée des designs graphiques professionnels, des illustrations et pictos originaux. "
        "Je maîtrise la typographie, la hiérarchie visuelle et les mises en page print et digital. "
        "Je maîtrise la suite Adobe Photoshop Illustrator InDesign et Figma. "
        "Je développe mon sens esthétique, je pratique les arts visuels et j'aime le dessin."
    ),
    # Blocs B05 + B07 — Communication et Marketing
    "communication": (
        "Je maîtrise la communication corporate et la stratégie de communication interne et externe. "
        "Je gère les relations publiques, les relations avec la presse et les situations de crise médiatique. "
        "Je maîtrise le storytelling et la narration pour captiver les audiences. "
        "Je rédige des communiqués de presse et dossiers de presse impactants. "
        "Je gère des communautés en ligne et anime les réseaux sociaux. "
        "Je crée du contenu digital engageant vidéos posts stories. "
        "Je développe des stratégies d'influence marketing et partenariats créateurs. "
        "J'optimise le référencement SEO et SEA et gère des campagnes publicitaires en ligne."
    ),
    # Bloc B08 — Data Analysis
    "data_analysis": (
        "Je programme en Python ou R pour l'analyse de données avec Pandas NumPy. "
        "Je maîtrise SQL pour interroger et manipuler des bases de données. "
        "Je crée des visualisations de données avec Matplotlib Seaborn Tableau Power BI Plotly. "
        "J'applique des statistiques descriptives et inférentielles. "
        "Je nettoie et prépare des données brutes pour l'analyse. "
        "Je crée des dashboards interactifs pour le pilotage décisionnel. "
        "Je réalise des analyses exploratoires EDA et interprète les résultats statistiques."
    ),
    # Bloc B09 — Machine Learning et IA
    "ml": (
        "Je développe et entraîne des algorithmes de machine learning supervisé et non supervisé. "
        "Je conçois des réseaux de neurones profonds pour le deep learning. "
        "J'implémente des modèles de traitement du langage naturel NLP. "
        "Je construis des modèles prédictifs pour la classification et la régression. "
        "Je maîtrise les frameworks TensorFlow PyTorch scikit-learn Keras HuggingFace BERT. "
        "Je déploie des modèles d'IA en production avec MLOps. "
        "J'optimise les hyperparamètres et évalue les performances des modèles."
    ),
    # Bloc B10 — Développement et Infrastructure
    "dev": (
        "Je programme en Python Java JavaScript avec les bonnes pratiques. "
        "Je développe des API REST robustes et scalables avec Flask Django FastAPI. "
        "Je gère des bases de données relationnelles PostgreSQL MySQL et NoSQL MongoDB. "
        "J'implémente des pipelines DevOps CI/CD avec Jenkins GitLab. "
        "Je déploie des infrastructures cloud AWS Azure GCP. "
        "J'utilise Git Docker et Kubernetes pour la conteneurisation et l'orchestration."
    ),
    # Bloc B11 — Ingénierie Technique
    "engineering": (
        "Je conçois des systèmes mécaniques complexes et innovants. "
        "J'applique les principes de thermodynamique et énergétique. "
        "Je maîtrise l'électronique et l'électrotechnique industrielle. "
        "J'utilise les logiciels de CAO DAO AutoCAD SolidWorks CATIA. "
        "Je développe et programme des systèmes embarqués. "
        "J'automatise des processus industriels et optimise la production."
    ),
}


@st.cache_data
def get_domaine_embeddings() -> dict:
    # Encodage des phrases de domaine au démarrage, mis en cache Streamlit.
    # Ces vecteurs servent de base à la pondération Likert dans build_profile_embedding.
    # Les recalculer à chaque analyse serait inutile — les phrases ne changent pas.
    bi_model = get_models()
    return {
        key: bi_model.encode(phrase, convert_to_tensor=True)
        for key, phrase in DOMAINE_PHRASES.items()
    }


def build_profile_embedding(levels: Dict[str, str], extra_text: str = ""):
    # Construction du vecteur profil par moyenne pondérée des embeddings de domaine.
    #
    # Principe : chaque domaine a un vecteur e_i et un poids w_i (issu de LIKERT_WEIGHTS).
    # Le vecteur profil est la somme pondérée divisée par la somme des poids :
    #   v_profil = somme(w_i * e_i) / somme(w_i)
    #
    # Si l'utilisateur a renseigné du texte libre (projet, outils, objectifs...),
    # on le fusionne avec un poids de 30% pour enrichir le signal sans écraser
    # la pondération Likert qui reste la source principale.
    #
    # La normalisation L2 finale garantit que semantic_search calcule bien
    # une similarité cosinus dans [0, 1] et non un produit scalaire brut.

    domaine_embeddings = get_domaine_embeddings()
    bi_model           = get_models()

    weighted_sum = None
    total_weight = 0.0

    for key, level in levels.items():
        weight = LIKERT_WEIGHTS.get(level, 0.0)
        if weight == 0.0:
            continue
        if key not in domaine_embeddings:
            continue

        emb = domaine_embeddings[key].float()
        weighted_sum = weight * emb if weighted_sum is None else weighted_sum + weight * emb
        total_weight += weight

    # Si tous les niveaux sont à Débutant, on bascule en mode texte libre pur.
    if weighted_sum is None or total_weight == 0.0:
        if extra_text.strip():
            return bi_model.encode(extra_text, convert_to_tensor=True)
        raise ValueError("Aucun niveau renseigné et pas de texte libre fourni.")

    profile_emb = weighted_sum / total_weight

    # Fusion 70/30 avec le texte libre
    if extra_text.strip():
        extra_emb   = bi_model.encode(extra_text, convert_to_tensor=True).float()
        profile_emb = 0.7 * profile_emb + 0.3 * extra_emb

    # Normalisation L2
    norm = torch.norm(profile_emb)
    if norm > 0:
        profile_emb = profile_emb / norm

    return profile_emb


def likert_to_semantic_text(levels: Dict[str, str]) -> str:
    # Génère un résumé textuel du profil à partir des niveaux Likert.
    # Sert principalement à construire le contexte des prompts Gemini
    # dans genai_augmentation — pas utilisé pour l'encodage NLP.
    parts = []
    for key, level in levels.items():
        weight = LIKERT_WEIGHTS.get(level, 0.0)
        if weight == 0.0:
            continue
        phrase = DOMAINE_PHRASES.get(key, "")
        if phrase:
            parts.append(f"[{level}] {phrase}")
    return " ".join(parts)


def analyze_profile(
        user_emb,
        bi_model: SentenceTransformer,
        comp_ids: List[str],
        comp_embeddings: np.ndarray,
        cross_model,        # paramètre conservé pour rétrocompatibilité, non utilisé
        comp_index: Dict,
        top_k: int = None
) -> Dict[str, float]:
    # Calcul de la similarité cosinus entre le vecteur profil et chaque compétence
    # du référentiel. On analyse toutes les compétences (top_k=None) pour ne rien
    # rater — le coût est négligeable vu que les embeddings sont déjà en mémoire.
    k    = len(comp_ids) if top_k is None else top_k
    hits = util.semantic_search(user_emb, comp_embeddings, top_k=k)

    candidate_indices = [hit['corpus_id'] for hit in hits[0]]
    candidate_ids     = [comp_ids[idx]    for idx in candidate_indices]
    raw_scores        = [hit['score']     for hit in hits[0]]

    print(f"\n DEBUG analyze_profile:")
    print(f"  - Scores cosinus bruts: min={min(raw_scores):.3f}, max={max(raw_scores):.3f}")
    print(f"  - Compétences analysées: {len(raw_scores)}/{len(comp_ids)}")

    # Clamp à 0 pour éviter des scores négatifs sur les compétences très éloignées.
    result = {
        candidate_ids[idx]: float(max(0.0, raw_scores[idx]))
        for idx in range(len(candidate_ids))
    }

    top_results = sorted(result.items(), key=lambda x: -x[1])[:5]
    print(f"  - Top 5 compétences matchées :")
    for cid, score in top_results:
        print(f"    * {comp_index[cid]['texte']}: {score:.2%}")

    return result


def recommend_jobs(
    comp_scores: Dict[str, float],
    data: Dict,
    top_n: int = 3
) -> Tuple[List[Dict], Dict[str, float]]:
    # Remontée des scores compétences -> blocs -> métiers en trois étapes.
    #
    # 1. On moyenne les scores de toutes les compétences d'un même bloc
    #    pour obtenir un score représentatif par bloc.
    #
    # 2. Pour chaque métier, on calcule une moyenne pondérée par rang des blocs requis.
    #    L'ordre dans blocs_requis encode la priorité : rang 0 = bloc cœur du métier.
    #    On utilise une décroissance exponentielle 1/2^rang — robuste quel que soit
    #    le nombre de blocs et insensible à l'ordre arbitraire dans le JSON.
    #
    #    Distribution effective :
    #      2 blocs → [67%, 33%]
    #      3 blocs → [57%, 29%, 14%]
    #      4 blocs → [53%, 27%, 13%, 7%]
    #
    # 3. Un bonus basé sur les compétences-clés du métier affine le classement
    #    entre métiers qui partagent exactement les mêmes blocs. Il pèse 30% du score final.

    comp_to_bloc = {}
    bloc_index   = {}

    for bloc in data['blocs']:
        bloc_id = bloc['id']
        bloc_index[bloc_id] = bloc
        for comp in bloc['competences']:
            comp_to_bloc[comp['id']] = bloc_id

    bloc_scores_raw = {b['id']: [] for b in data['blocs']}
    for cid, score in comp_scores.items():
        if cid in comp_to_bloc:
            bloc_scores_raw[comp_to_bloc[cid]].append(score)

    bloc_scores = {
        bid: float(np.mean(scores)) if scores else 0.0
        for bid, scores in bloc_scores_raw.items()
    }

    recommendations = []
    for job in data['metiers']:
        bloc_ids = job.get('blocs_requis', [])
        if not bloc_ids:
            continue

        # Pondération exponentielle 1/2^rang — se normalise automatiquement
        weights      = [1.0 / (2 ** rank) for rank in range(len(bloc_ids))]
        total_weight = sum(weights)
        weighted_sum = sum(
            bloc_scores.get(bid, 0.0) * w
            for bid, w in zip(bloc_ids, weights)
        )
        base_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        cles  = job.get('competences_cles', [])
        bonus = float(np.mean([comp_scores.get(cid, 0.0) for cid in cles])) * 0.3 if cles else 0.0

        job_score = base_score * 0.7 + bonus

        job_blocs = [
            {'nom': bloc_index[bid]['nom'], 'score': bloc_scores.get(bid, 0.0)}
            for bid in bloc_ids
            if bid in bloc_index
        ]

        recommendations.append({
            'titre':       job['titre'],
            'description': job.get('description', ''),
            'filiere':     job.get('filiere', ''),
            'secteurs':    job.get('secteurs', []),
            'score':       job_score,
            'blocs':       job_blocs
        })

    top_all = sorted(recommendations, key=lambda x: x['score'], reverse=True)[:5]
    print(f"\n DEBUG recommend_jobs (top 5) :")
    for r in top_all:
        print(f"  {r['titre']:45s} -> {r['score']:.2%}")

    top_jobs = sorted(recommendations, key=lambda x: x['score'], reverse=True)[:top_n]
    return top_jobs, bloc_scores