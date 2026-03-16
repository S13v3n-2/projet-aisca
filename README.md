# AISCA - Agent Intelligent de Cartographie des Competences

Projet realise dans le cadre du module IA Generative en M1 Data Engineering & AI a l'EFREI.

Le projet est un outil web qui analyse les competences de quelqu'un a partir d'un questionnaire et qui lui recommande des metiers adaptes a son profil.

## Fonctionnement

Le pipeline est assez simple au final :

1. L'utilisateur remplit un questionnaire hybride (echelle likert pour les niveaux par domaine + texte libre pour decrire ses experiences et aspirations)
2. Si le texte libre est trop court, on l'enrichit via Gemini pour avoir plus de matiere semantique
3. On genere un embedding du profil complet avec SBERT (paraphrase-multilingual-mpnet-base-v2, c'est le seul modele multilingue qui marchait bien sur du francais)
4. On compare par similarite cosinus avec les embeddings pre-calcules de toutes les competences du referentiel
5. On agrege les scores par bloc de competences avec une moyenne ponderee par rang (le premier bloc requis d'un metier pese plus que le troisieme)
6. On recommande les 3 metiers les plus adaptés
7. Gemini genere un plan de progression personnalise et une bio professionnelle

Le referentiel contient 11 blocs de competences et 40 metiers, chaque metier est lie a 2-4 blocs requis.

## Stack technique

- Python 3.10+
- Streamlit pour l'interface web
- Sentence-Transformers (paraphrase-multilingual-mpnet-base-v2) pour les embeddings
- Google Gemini 2.5 Flash pour l'enrichissement de texte et la generation
- Plotly pour les graphiques (radar chart, bar chart)
- NumPy pour le scoring

## Installation

```bash
git clone https://github.com/S13v3n-2/projet-aisca.git
cd projet-aisca
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

Creer un fichier `.env` dans le dossier `src/` :

```
GEMINI_API_KEY=ta_cle_ici
```

Lancer l'application :

```bash
cd src
streamlit run test-visualisations.py
```

## Structure du projet

```
projet-aisca/
    data/
        referentiel.json        # referentiel competences et metiers
        data.py                 # donnees en dur (doublon du json)
    src/
        scoring.py              # logique NLP et scoring
        genai_augmentation.py   # appels Gemini + cache
        test-visualisations.py  # interface Streamlit principale
        visualisations.py       # ancienne version de l'interface
        scoring_save.py         # ancien scoring avec cross-encoder
        app.py                  # vide (prevu pour Flask)
        visuel/                 # images et assets
    tests/
        test_gpu.py             # test detection GPU
    .env                        # cle API (pas versionne)
    requirements.txt
```

## Ce qu'on aurait aime ameliorer

- Ajouter un vrai systeme de comptes utilisateurs pour sauvegarder les profils
- Tester d'autres modeles d'embedding (e5-multilingual, BGE) pour comparer les resultats
- Faire une interface plus clean avec un vrai step wizard au lieu du long formulaire a scroller
- Ajouter des tests unitaires serieux sur le scoring
- Mettre en place un referentiel dynamique editable depuis l'interface au lieu du JSON en dur
- Deployer sur un serveur pour que ce soit accessible en ligne

## Auteurs

Steven - [@S13v3n-2](https://github.com/S13v3n-2)
Ilies - [@IliesIdir](https://github.com/IliesIdir)
