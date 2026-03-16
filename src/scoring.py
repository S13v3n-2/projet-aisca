# module de scoring sémantique pour matcher les compétences avec les métiers
# on utilise SBERT (bi-encoder multilingue) parce que c'est le seul qui gère
# correctement le français sans devoir traduire les textes avant

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer, util


# on charge le modèle une seule fois grâce au cache streamlit
# sinon ça prenait 10 secondes à chaque interaction, pas ouf
@st.cache_resource
def get_models() -> SentenceTransformer:
    bi_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    return bi_model


# on charge le référentiel et on pré-calcule tous les embeddings des compétences
# comme ça au moment de l'analyse c'est instantané
@st.cache_data
def load_and_index_data(json_path: str) -> Tuple[Dict, Dict, List[str], np.ndarray]:
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

    bi_model = get_models()
    comp_ids = list(comp_index.keys())
    texts = [comp_index[cid]['texte'] for cid in comp_ids]

    # batch_size=64 ça passe bien même sur CPU, on a testé
    embeddings = bi_model.encode(
        texts,
        convert_to_tensor=True,
        show_progress_bar=False,
        batch_size=64
    )

    return data, comp_index, comp_ids, embeddings


# on répète les phrases selon le niveau déclaré par l'utilisateur
# l'idée c'est que si quelqu'un se dit expert en ML, on veut que ça pèse
# beaucoup plus dans l'embedding final que quelqu'un qui se dit intermédiaire
# on a testé plusieurs valeurs et 3x/5x ça donne les meilleurs résultats
# pour les débutants on met 0 parce que sinon ça ajoutait du bruit inutile
LIKERT_REPETITIONS = {
    "Débutant":      0,
    "Intermédiaire": 1,
    "Avancé":        3,
    "Expert":        5,
}

# phrases sémantiques pour chaque domaine, on reprend le vocabulaire exact
# du référentiel pour que le matching soit le plus précis possible
# au début on avait des phrases plus courtes mais les scores étaient trop bas
DOMAINE_PHRASES = {
    # vocabulaire calqué sur le bloc B02 du référentiel
    "business": (
        "Je maîtrise la stratégie d'entreprise et la définition de business models innovants. "
        "Je réalise des analyses de marché, des études de faisabilité technique et financière. "
        "Je pilote des projets complexes avec méthodes agiles, je conduis des analyses concurrentielles "
        "et j'accompagne le changement et les transformations organisationnelles."
    ),
    # vocabulaire du bloc B01, on l'utilise surtout pour le profil test Juriste
    "juridique": (
        "Je maîtrise le droit des affaires, le droit commercial et le droit du travail. "
        "J'analyse et interprète des contrats et textes juridiques complexes. "
        "Je rédige des actes juridiques et documents légaux professionnels. "
        "J'assure la conformité réglementaire RGPD et le respect des normes. "
        "J'effectue une veille réglementaire et juridique continue. "
        "Je protège la propriété intellectuelle et gère les procédures de contentieux."
    ),
    # bloc B04
    "finance": (
        "Je maîtrise la comptabilité générale et analytique en normes françaises et IFRS. "
        "Je réalise des analyses financières et diagnostics économiques approfondis. "
        "J'élabore et suis les budgets prévisionnels et contrôle les écarts. "
        "Je crée des tableaux de bord financiers pour le pilotage décisionnel. "
        "Je gère la trésorerie et évalue la rentabilité des investissements."
    ),
    # bloc B06
    "design": (
        "Je maîtrise le design graphique et la création d'identité visuelle complète. "
        "J'assure la direction artistique de projets créatifs complexes. "
        "Je crée des designs graphiques professionnels, des illustrations et pictos originaux. "
        "Je maîtrise la typographie, la hiérarchie visuelle et les mises en page print et digital. "
        "Je maîtrise la suite Adobe Photoshop Illustrator InDesign et Figma. "
        "Je développe mon sens esthétique, je pratique les arts visuels et j'aime le dessin."
    ),
    # blocs B05 + B07
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
    # bloc B08
    "data_analysis": (
        "Je programme en Python ou R pour l'analyse de données avec Pandas NumPy. "
        "Je maîtrise SQL pour interroger et manipuler des bases de données. "
        "Je crée des visualisations de données avec Matplotlib Seaborn Tableau Power BI Plotly. "
        "J'applique des statistiques descriptives et inférentielles. "
        "Je nettoie et prépare des données brutes pour l'analyse. "
        "Je crée des dashboards interactifs pour le pilotage décisionnel. "
        "Je réalise des analyses exploratoires EDA et interprète les résultats statistiques."
    ),
    # bloc B09
    "ml": (
        "Je développe et entraîne des algorithmes de machine learning supervisé et non supervisé. "
        "Je conçois des réseaux de neurones profonds pour le deep learning. "
        "J'implémente des modèles de traitement du langage naturel NLP. "
        "Je construis des modèles prédictifs pour la classification et la régression. "
        "Je maîtrise les frameworks TensorFlow PyTorch scikit-learn Keras HuggingFace BERT. "
        "Je déploie des modèles d'IA en production avec MLOps. "
        "J'optimise les hyperparamètres et évalue les performances des modèles."
    ),
    # bloc B10
    "dev": (
        "Je programme en Python Java JavaScript avec les bonnes pratiques. "
        "Je développe des API REST robustes et scalables avec Flask Django FastAPI. "
        "Je gère des bases de données relationnelles PostgreSQL MySQL et NoSQL MongoDB. "
        "J'implémente des pipelines DevOps CI/CD avec Jenkins GitLab. "
        "Je déploie des infrastructures cloud AWS Azure GCP. "
        "J'utilise Git Docker et Kubernetes pour la conteneurisation et l'orchestration."
    ),
    # bloc B11
    "engineering": (
        "Je conçois des systèmes mécaniques complexes et innovants. "
        "J'applique les principes de thermodynamique et énergétique. "
        "Je maîtrise l'électronique et l'électrotechnique industrielle. "
        "J'utilise les logiciels de CAO DAO AutoCAD SolidWorks CATIA. "
        "Je développe et programme des systèmes embarqués. "
        "J'automatise des processus industriels et optimise la production."
    ),
}


# prend les niveaux likert de l'utilisateur et les transforme en texte sémantique
# on répète les phrases selon le niveau pour que le bi-encoder capte la pondération
def likert_to_semantic_text(levels: Dict[str, str]) -> str:
    parts = []
    for key, level in levels.items():
        repetitions = LIKERT_REPETITIONS.get(level, 0)
        if repetitions == 0:
            continue  # débutant = on skip, ça pollue l'embedding sinon
        phrase = DOMAINE_PHRASES.get(key, "")
        if phrase:
            for _ in range(repetitions):
                parts.append(phrase)

    return " ".join(parts)


# analyse le profil utilisateur et renvoie les scores de match par compétence
# on utilise uniquement le bi-encoder maintenant, le cross-encoder anglais
# marchait vraiment pas sur du texte français donc on l'a viré
# on garde le paramètre cross_model pour pas casser l'interface mais il sert à rien
def analyze_profile(
        user_text: str,
        bi_model: SentenceTransformer,
        comp_ids: List[str],
        comp_embeddings: np.ndarray,
        cross_model,           # pas utilisé, on garde pour compatibilité
        comp_index: Dict,
        top_k: int = None
) -> Dict[str, float]:
    user_emb = bi_model.encode(user_text, convert_to_tensor=True)

    # on analyse toutes les compétences (top_k=None), au début on avait top_k=50
    # mais ça coupait des compétences pertinentes, du coup on analyse tout
    k = len(comp_ids) if top_k is None else top_k
    hits = util.semantic_search(user_emb, comp_embeddings, top_k=k)

    candidate_indices = [hit['corpus_id'] for hit in hits[0]]
    candidate_ids     = [comp_ids[idx] for idx in candidate_indices]
    raw_scores        = [hit['score']   for hit in hits[0]]

    print(f"\n DEBUG analyze_profile:")
    print(f"  - Scores cosinus bruts: min={min(raw_scores):.3f}, max={max(raw_scores):.3f}")
    print(f"  - Compétences analysées: {len(raw_scores)}/{len(comp_ids)}")

    # on garde les scores cosinus bruts, on avait essayé une normalisation min-max
    # mais ça faussait les résultats en écrasant les écarts entre compétences
    result = {
        candidate_ids[idx]: float(max(0.0, raw_scores[idx]))
        for idx in range(len(candidate_ids))
    }

    top_results = sorted(result.items(), key=lambda x: -x[1])[:5]
    print(f"  - Top 5 compétences matchées :")
    for cid, score in top_results:
        print(f"    * {comp_index[cid]['texte']}: {score:.2%}")

    return result


# recommande les métiers les plus adaptés au profil
# on fait une moyenne pondérée par rang des blocs requis pour chaque métier
# le premier bloc requis compte 4x plus que le troisième, ça permet de mieux
# discriminer entre des métiers qui partagent les mêmes blocs
def recommend_jobs(
    comp_scores: Dict[str, float],
    data: Dict,
    top_n: int = 3
) -> Tuple[List[Dict], Dict[str, float]]:
    # poids par rang, on a itéré plusieurs fois sur ces valeurs
    # 4/2/1/0.5 ça donne un bon équilibre entre précision et diversité
    RANK_WEIGHTS = [4.0, 2.0, 1.0, 0.5]

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

        # moyenne pondérée par rang
        total_weight = 0.0
        weighted_sum = 0.0
        for rank, bid in enumerate(bloc_ids):
            weight = RANK_WEIGHTS[rank] if rank < len(RANK_WEIGHTS) else 0.25
            weighted_sum += bloc_scores.get(bid, 0.0) * weight
            total_weight += weight
        base_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # bonus compétences clés (30% du score final)
        # ça affine le classement entre métiers qui ont les mêmes blocs
        cles = job.get('competences_cles', [])
        bonus = float(np.mean([comp_scores.get(cid, 0.0) for cid in cles])) * 0.3 if cles else 0.0

        job_score = base_score * 0.7 + bonus

        job_blocs = [
            {'nom': bloc_index[bid]['nom'], 'score': bloc_scores.get(bid, 0.0)}
            for bid in bloc_ids
        ]

        recommendations.append({
            'titre':       job['titre'],
            'description': job.get('description', ''),
            'score':       job_score,
            'blocs':       job_blocs
        })

    top_all = sorted(recommendations, key=lambda x: x['score'], reverse=True)[:5]
    print(f"\n DEBUG recommend_jobs (top 5) :")
    for r in top_all:
        print(f"  {r['titre']:45s} -> {r['score']:.2%}")

    top_jobs = sorted(recommendations, key=lambda x: x['score'], reverse=True)[:top_n]
    return top_jobs, bloc_scores
