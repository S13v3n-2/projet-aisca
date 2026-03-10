# 🧠 AISCA — Agent Intelligent Sémantique pour la Cartographie des Compétences

> Projet d'IA générative réalisé dans le cadre d'études en intelligence artificielle.

---

## Description

**AISCA** est un agent intelligent basé sur le traitement du langage naturel (NLP) et l'IA générative, conçu pour analyser, extraire et cartographier automatiquement des compétences à partir de textes (CVs, offres d'emploi, référentiels de formation, etc.).

Le projet vise à automatiser la mise en correspondance sémantique entre des profils et des compétences, en s'appuyant sur des modèles de langage et des techniques d'embedding vectoriel.

---

## Structure du projet

```
projet-aisca/
├── data/               # Données brutes et traitées (CVs, offres, référentiels)
├── docs/               # Documentation technique et rapports
├── src/                # Code source principal de l'agent
├── tests/              # Tests unitaires et d'intégration
├── .gitignore
├── requirements.txt    # Dépendances Python
└── README.md
```

---

## Fonctionnalités

-  **Extraction de compétences** depuis des textes non structurés (CVs, fiches de poste)
-  **Analyse sémantique** par embeddings vectoriels pour mesurer la similarité entre compétences
-  **Cartographie des compétences** sous forme de graphe ou de clusters thématiques
-  **IA générative** pour enrichir, reformuler ou suggérer des compétences manquantes
-  **Visualisation** des relations entre compétences et profils

---

## 🛠 Technologies utilisées

| Catégorie | Outils |
|---|---|
| Langage | Python 3.10+ |
| IA / NLP | Transformers, Sentence-Transformers, OpenAI API / Ollama |
| Vectorisation | FAISS, ChromaDB |
| Visualisation | Matplotlib, NetworkX, Plotly |
| Notebooks | Jupyter Notebook |
| Versioning | Git / GitHub |

---

##  Installation

### Prérequis

- Python 3.10 ou supérieur
- `pip` ou `conda`

### Étapes

```bash
# 1. Cloner le dépôt
git clone https://github.com/S13v3n-2/projet-aisca.git
cd projet-aisca

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

### Variables d'environnement (si applicable)

Créer un fichier `.env` à la racine :

```env
OPENAI_API_KEY=your_api_key_here
```

---

##  Utilisation

```bash
# Lancer le notebook principal
jupyter notebook

# Ou exécuter le script principal
python src/main.py
```

---

##  Tests

```bash
pytest tests/
```

---

##  Données

Les données utilisées dans ce projet sont placées dans le dossier `data/`. Elles peuvent inclure :
- Des exemples de CVs anonymisés
- Des offres d'emploi
- Des référentiels de compétences (ex. ESCO, ROME)

> ⚠ Les données sensibles ou personnelles ne sont pas versionnées (voir `.gitignore`).

---

##  Contexte académique

Ce projet a été réalisé dans le cadre d'un cursus en **Intelligence Artificielle**. Il explore les applications de l'IA générative et du NLP pour répondre à un besoin concret : la gestion et la valorisation des compétences.

---

##  Auteur

**Steven** — [@S13v3n-2](https://github.com/S13v3n-2)  
**Ilies** — [IliesIdir](https://github.com/IliesIdir)
---

##  Licence

Ce projet est à usage académique. Tous droits réservés.
