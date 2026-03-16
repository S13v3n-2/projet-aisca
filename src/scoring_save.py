# ancienne version du scoring, on la garde comme backup
# c'est la version qui utilisait le cross-encoder anglais (ms-marco)
# on l'a remplacée par scoring.py qui utilise seulement le bi-encoder multilingue
# parce que le cross-encoder anglais marchait pas du tout sur du texte français

import json
import os
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer, util, CrossEncoder
"""
import google.generativeai as genai
"""


# charge le référentiel et indexe tout pour un accès rapide
def charger_referentiel(chemin):
    with open(chemin, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # on indexe par id pour avoir du O(1) au lieu de parcourir des listes
    comp_index = {c['id']: c for b in data['blocs'] for c in b['competences']}
    metier_index = {m['id']: m for m in data['metiers']}
    bloc_index = {b['id']: b for b in data['blocs']}
    return data, comp_index, metier_index, bloc_index


# initialise le bi-encoder et pré-calcule les vecteurs
# on utilisait all-MiniLM-L6-v2 ici, c'est un modèle anglais
# dans la nouvelle version on est passé sur paraphrase-multilingual pour le français
def initialiser_moteur_vectoriel(comp_index):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    ids = list(comp_index.keys())
    textes = [c['texte'] for c in comp_index.values()]
    embeddings = model.encode(textes, convert_to_tensor=True)

    vector_store = {ids[i]: embeddings[i] for i in range(len(ids))}
    return model, vector_store


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# analyse le profil avec bi-encoder + cross-encoder (ancienne approche)
# le cross-encoder re-ranke les top 20 du bi-encoder
# ça marchait bien en anglais mais en français les scores étaient n'importe quoi
def analyser_profil(user_input, model_bi, vector_store, model_cross, comp_index):
    # bi-encoder : on compare le texte user avec toutes les compétences
    user_emb = model_bi.encode(user_input, convert_to_tensor=True)
    initial_results = []
    for cid, cemb in vector_store.items():
        score = util.cos_sim(user_emb, cemb).item()
        initial_results.append({"id": cid, "score": score})

    # on garde le top 20 pour le reranking
    top_20 = sorted(initial_results, key=lambda x: x['score'], reverse=True)[:20]
    passages = [comp_index[res['id']]['texte'] for res in top_20]

    print("\nTop 20 des competences (embedding)")
    for top in top_20:
        print(f"{top['id']} : {comp_index[top['id']]['texte']} ({top['score']:.2%})")

    # cross-encoder : re-ranking plus précis mais en anglais seulement
    # c'est pour ça qu'on a abandonné cette approche
    ranks = model_cross.rank(user_input, passages, return_documents=True)

    final_scores = {}
    for r in ranks:
        real_id = top_20[r['corpus_id']]['id']
        # sigmoid pour normaliser entre 0 et 1
        final_scores[real_id] = sigmoid(r['score'])

    print("\nApres cross encoder")
    for cid, score in final_scores.items():
        print(f"{cid} : {comp_index[cid]['texte']} ({score:.2%})")
    return final_scores


# recommandation de métiers basée sur les scores par bloc
# version simple sans pondération par rang (la nouvelle version dans scoring.py fait mieux)
def recommander_metiers(scores_comp, data):
    bloc_scores = {b['id']: [] for b in data['blocs']}
    comp_to_bloc = {c['id']: b['id'] for b in data['blocs'] for c in b['competences']}

    for cid, score in scores_comp.items():
        if cid in comp_to_bloc:
            bloc_scores[comp_to_bloc[cid]].append(score)

    avg_bloc_scores = {bid: np.mean(s) if s else 0.0 for bid, s in bloc_scores.items()}

    recommandations = []
    for metier in data['metiers']:
        score_m = 0
        poids_total = 0
        for bid in metier['blocs_requis']:
            poids = 1.0
            score_m += avg_bloc_scores.get(bid, 0) * poids
            poids_total += poids

        recommandations.append({
            "titre": metier['titre'],
            "score": score_m / poids_total if poids_total > 0 else 0
        })
    return sorted(recommandations, key=lambda x: x['score'], reverse=True)[:3], avg_bloc_scores


# cache pour les appels genai (commenté parce qu'on utilise le nouveau module maintenant)
CACHE_FILE = "aisca_cache.json"

"""def call_genai_with_cache(prompt):
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f: cache = json.load(f)

    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    if prompt_hash in cache: return cache[prompt_hash]

    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)

    cache[prompt_hash] = response.text
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)
    return response.text"""

# fonction de test pour lancer le pipeline complet
def test():
    data, comp_idx, metier_idx, bloc_idx = charger_referentiel("data/referentiel.json")
    bi_model, vstore = initialiser_moteur_vectoriel(comp_idx)
    cross_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

    user_query = "Je développe des scripts Python pour nettoyer des bases SQL et entraîner des modèles de classification."
    scores_c = analyser_profil(user_query, bi_model, vstore, cross_model, comp_idx)
    top_metiers, scores_blocs = recommander_metiers(scores_c, data)

    print("\nTop 3 des metiers recommandes")
    for index, metier in enumerate(top_metiers[:3]):
        print(f"Top {index + 1} Metier : {metier['titre']} ({metier['score']:.2%})")

# pareil mais avec un prompt custom, on s'en servait pour tester depuis l'interface
def call_model(prompt):
    data, comp_idx, metier_idx, bloc_idx = charger_referentiel("data/referentiel.json")
    bi_model, vstore = initialiser_moteur_vectoriel(comp_idx)
    cross_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

    user_query = prompt
    scores_c = analyser_profil(user_query, bi_model, vstore, cross_model, comp_idx)
    top_metiers, scores_blocs = recommander_metiers(scores_c, data)
    return top_metiers, scores_blocs
