# AISCA — Agent Intelligent Sémantique pour la Cartographie des Compétences

> Projet de fin de module IA Générative — M1 Data Engineering & AI, EFREI Paris

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)
![SBERT](https://img.shields.io/badge/SBERT-paraphrase--multilingual-green?style=flat-square)
![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-orange?style=flat-square&logo=google)
![Status](https://img.shields.io/badge/status-academic%20project-lightgrey?style=flat-square)

---

## Pourquoi AISCA ?

Les outils d'orientation professionnelle existants se limitent à du matching par mots-clés — ils ne comprennent pas vraiment ce que quelqu'un sait faire. AISCA est notre réponse à ce problème : un mini-agent RAG qui analyse sémantiquement un profil utilisateur et recommande les 3 métiers les plus adaptés, avec un plan de progression personnalisé généré par Gemini.

Le projet a été conçu et développé en binôme dans le cadre du module Generative AI à l'EFREI.

---

## Ce que fait AISCA

L'utilisateur remplit un questionnaire guidé (wizard Streamlit en 6 étapes), décrit ses compétences librement ou via des échelles Likert, et obtient en sortie :

- **3 métiers recommandés** issus d'un référentiel de 40 métiers répartis en 11 blocs de compétences
- **Un score de similarité** par métier, calculé par similarité cosinus sur embeddings SBERT
- **Un plan de progression personnalisé** + une bio professionnelle synthétique, générés par Gemini 2.5 Flash

---

## Pipeline technique

```
Questionnaire (6 étapes Streamlit)
        ↓
Enrichissement sémantique (Gemini — textes courts < 5 mots)
        ↓
Embedding profil (paraphrase-multilingual-mpnet-base-v2, 768 dims)
        ↓
Similarité cosinus vs référentiel pré-calculé
        ↓
Agrégation par bloc — pondération exponentielle par rang (4× → 0.5×)
        ↓
Coverage Score = Σ(Wᵢ × Sᵢ) / ΣWᵢ  →  Top 3 métiers
        ↓
Génération Gemini (1 appel API) — plan de progression + bio
```

---

## Stack

| Catégorie | Outils |
|---|---|
| Langage | Python 3.10+ |
| Interface | Streamlit |
| NLP / Embeddings | Sentence-Transformers (`paraphrase-multilingual-mpnet-base-v2`) |
| IA Générative | Google Gemini 2.5 Flash API |
| Visualisation | Plotly (radar chart + bar chart) |
| Optimisation | Caching JSON SHA256 — zéro appel redondant |
| Versioning | Git / GitHub |

---

## Structure du projet

```
projet-aisca/
    data/
        referentiel.json        # referentiel competences et metiers
    src/
        scoring.py              # logique NLP et scoring
        genai_augmentation.py   # appels Gemini + cache
        visualisations.py       # interface Streamlit 
        app.py                  # vide (prevu pour Flask)
    .env                        # cle API (pas versionne)
    requirements.txt
```

---

## Installation

**Prérequis :** Python 3.10+, pip

```bash
# 1. Cloner le dépôt
git clone https://github.com/S13v3n-2/projet-aisca.git
cd projet-aisca

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

Créer un fichier `.env` à la racine :

```env
GEMINI_API_KEY=your_api_key_here
```

---

## Lancement

```bash
streamlit run src/main.py
```

L'application s'ouvre dans le navigateur. Le wizard guide l'utilisateur en 6 étapes : domaines, niveaux, soft skills, expériences, outils, parcours.

---

## Tests

```bash
pytest tests/
```

---

## Données

Le référentiel (`data/referentiel.json`) structure 40 métiers en 11 blocs de compétences, chacun associé à une liste de compétences encodées. Chaque métier spécifie 2 à 4 blocs requis avec un rang de priorité.

Les embeddings du référentiel sont pré-calculés au premier lancement et mis en cache localement.

> ⚠️ Aucune donnée personnelle n'est versionnée — voir `.gitignore`.

---

## Limites connues

- Application non déployée en ligne (usage local uniquement)
- Référentiel JSON statique, non éditable depuis l'interface
- Pas de comptes utilisateurs ni sauvegarde de profils
- Évaluation des résultats manuelle — pas de benchmark automatisé
- Quota Gemini Free Tier parfois atteint en usage intensif

---

## Auteurs

**Steven** — [@S13v3n-2](https://github.com/S13v3n-2)  
**Ilies** — [@IliesIdir](https://github.com/IliesIdir)

---

## Contexte académique

Projet réalisé dans le cadre du module **Generative AI** — M1 Data Engineering & AI, EFREI Paris.  
Usage académique uniquement. Tous droits réservés.
