import json
import os
import hashlib
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util, CrossEncoder


# ÉTAPE 1 : CHARGEMENT ET INDEXATION
def charger_referentiel(chemin):
    with open(chemin, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Indexation pour accès O(1)
    comp_index = {c['id']: c for b in data['blocs'] for c in b['competences']} # Indexation des compétences
    metier_index = {m['id']: m for m in data['metiers']}                       # Indexation des métiers
    bloc_index = {b['id']: b for b in data['blocs']}                           # Indexation des blocs
    print(bloc_index)
    return data, comp_index, metier_index, bloc_index


# ÉTAPE 2 : MOTEUR SÉMANTIQUE (BI-ENCODER)
def initialiser_moteur_vectoriel(comp_index):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    ids = list(comp_index.keys())
    textes = [c['texte'] for c in comp_index.values()]
    embeddings = model.encode(textes, convert_to_tensor=True)

    # Association de chaque id de compétence a son vecteur
    vector_store = {ids[i]: embeddings[i] for i in range(len(ids))}
    return model, vector_store


# ÉTAPE 3 : LOGIQUE DE SCORING ET RERANKING
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def analyser_profil(user_input, model_bi, vector_store, model_cross, comp_index):
    # 1. Retrieval (Bi-Encoder)
    user_emb = model_bi.encode(user_input, convert_to_tensor=True)
    initial_results = []
    for cid, cemb in vector_store.items():
        score = util.cos_sim(user_emb, cemb).item()
        initial_results.append({"id": cid, "score": score})

    # Top 20 pour le reranking
    top_20 = sorted(initial_results, key=lambda x: x['score'], reverse=True)[:20]
    passages = [comp_index[res['id']]['texte'] for res in top_20]

    ## Visiualisation des meilleures compétences
    print("\n##################################### Top 20 des competences Embedding #####################################\n")
    for top in top_20:
        print(f"{top['id']} : {comp_index[top['id']]['texte']} ({top['score']:.2%})")


    # 2. Re-ranking (Cross-Encoder)
    # On récupère les rangs et scores précis
    ranks = model_cross.rank(user_input, passages, return_documents=True)

    final_scores = {}
    for r in ranks:
        real_id = top_20[r['corpus_id']]['id']
        # Normalisation entre 0 et 1 pour le scoring
        final_scores[real_id] = sigmoid(r['score'])
    print("\n##################################### Apres cross encoder #####################################\n")
    for cid, score in final_scores.items():
        print(f"{cid} : {comp_index[cid]['texte']} ({score:.2%})")
        #print(f"{cid} : {score}")
    return final_scores


# ÉTAPE 4 : RECOMMANDATION DE MÉTIERS
def recommander_metiers(scores_comp,data):
    # Agrégation par bloc
    bloc_scores = {b['id']: [] for b in data['blocs']}
    comp_to_bloc = {c['id']: b['id'] for b in data['blocs'] for c in b['competences']}

    for cid, score in scores_comp.items():
        if cid in comp_to_bloc:
            bloc_scores[comp_to_bloc[cid]].append(score)

    # Moyenne par bloc
    avg_bloc_scores = {bid: np.mean(s) if s else 0.0 for bid, s in bloc_scores.items()}

    recommandations = []
    for metier in data['metiers']:
        score_m = 0
        poids_total = 0
        for bid in metier['blocs_requis']:

            poids = 1.0  # On peut ajouter un poids spécifique dans le JSON
            score_m += avg_bloc_scores.get(bid, 0) * poids
            poids_total += poids

        recommandations.append({
            "titre": metier['titre'],
            "score": score_m / poids_total if poids_total > 0 else 0
        })
    return sorted(recommandations, key=lambda x: x['score'], reverse=True)[:3], avg_bloc_scores


#  ÉTAPE 5 : AGENT RAG AVEC CACHING
CACHE_FILE = "aisca_cache.json"


"""def call_genai_with_cache(prompt):
    # Gestion du cache local [cite: 71]
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f: cache = json.load(f)

    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    if prompt_hash in cache: return cache[prompt_hash]

    # Appel API (Gemini recommandé) [cite: 60]
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)

    cache[prompt_hash] = response.text
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)
    return response.text"""


# === EXÉCUTION DU PIPELINE ===
# Récupère le dossier où se trouve le fichier scoring.py (src/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construit le chemin vers data/referentiel.json
# On remonte d'un cran (..) puis on descend dans data/
chemin_json = os.path.join(BASE_DIR, "..", "data", "referentiel.json")
# Initialisation
data, comp_idx, metier_idx, bloc_idx = charger_referentiel(chemin_json)
bi_model, vstore = initialiser_moteur_vectoriel(comp_idx)
cross_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

# Test utilisateur
user_query = "Je développe des scripts Python pour nettoyer des bases SQL et entraîner des modèles de classification."
scores_c = analyser_profil(user_query, bi_model, vstore, cross_model, comp_idx)
top_metiers, scores_blocs = recommander_metiers(scores_c,data)

print("\n##################################### Top 3 des métiers recommandés #####################################\n")
for index, metier in enumerate(top_metiers[:3]):
    print(f"Top {index + 1} Métier : {metier['titre']} ({metier['score']:.2%})")