# data.py

# 1. RÉFÉRENTIEL DES BLOCS DE COMPÉTENCES
# Ce dictionnaire sert à calculer les scores de similarité (Si) avec SBERT.
# Chaque clé est un "Bloc" et la valeur est la liste des phrases vecteurs.

COMPETENCY_BLOCKS = {
    # --- FAC DE DROIT ---
    "Juridique_Reglementaire": [
        "Analyse et rédaction de contrats juridiques",
        "Veille juridique et réglementaire stricte",
        "Connaissance du droit des affaires et commercial",
        "Application du droit du travail et relations sociales",
        "Gestion des contentieux et procédures judiciaires",
        "Protection des données personnelles et RGPD"
    ],

    # --- ÉCOLE DE COMMERCE ---
    "Business_Strategie": [
        "Analyse financière et élaboration de business plan",
        "Définition de la stratégie d'entreprise et développement",
        "Négociation commerciale complexe",
        "Gestion de projet transversal",
        "Analyse de marché et veille concurrentielle",
        "Management et leadership d'équipe"
    ],
    "Marketing_Vente": [
        "Marketing digital, SEO et réseaux sociaux",
        "Étude de marché et analyse du comportement consommateur",
        "Techniques de vente et gestion de la relation client (CRM)",
        "Stratégie de branding et positionnement de marque",
        "Fidélisation client et expérience utilisateur",
        "Stratégie e-commerce et parcours omnicanal"
    ],
    "Finance_Comptabilite": [
        "Comptabilité générale et analytique",
        "Contrôle de gestion, reporting et KPI",
        "Analyse financière et valorisation d'entreprise",
        "Gestion de trésorerie et budget",
        "Audit financier et procédures de contrôle",
        "Fiscalité d'entreprise et déclarations"
    ],

    # --- ÉCOLE DE COMMUNICATION ---
    "Communication_Medias": [
        "Stratégie de communication globale et plan média",
        "Relations presse et relations publiques",
        "Gestion de la communication de crise",
        "Rédaction de contenus et storytelling",
        "Communication interne et culture d'entreprise",
        "Organisation d'événements et activation de marque"
    ],
    "Creation_Design": [
        "Direction artistique et concepts visuels",
        "Conception graphique, logo et identité visuelle",
        "Production audiovisuelle et montage vidéo",
        "Design d'interface UX et expérience utilisateur UI",
        "Photographie professionnelle et retouche image",
        "Motion design et animation 2D/3D"
    ],
    "Digital_Social": [
        "Community management et animation de communautés",
        "Stratégie social media et calendrier éditorial",
        "Référencement naturel SEO et payant SEA",
        "Content marketing et stratégie de contenu",
        "Marketing d'influence et gestion de partenariats",
        "Analytics, suivi de trafic et mesure de performance"
    ],

    # --- ÉCOLE D'INGÉNIEUR ---
    "Data_Analysis": [
        "Nettoyage et préparation de données (Data Cleaning)",
        "Visualisation de données et storytelling (DataViz)",
        "Analyse statistique descriptive et inférentielle",
        "Manipulation de bases de données avec SQL et Python",
        "Création de tableaux de bord et reporting automatisé"
    ],
    "Machine_Learning_IA": [
        "Entraînement de modèles de classification et régression",
        "Deep Learning et réseaux de neurones artificiels",
        "Évaluation de modèles (Précision, Recall, F1-Score)",
        "Traitement du langage naturel (NLP) et analyse de texte",
        "Vision par ordinateur (Computer Vision)"
    ],
    "Dev_Infrastructure": [
        "Programmation orientée objet en Python, Java ou C++",
        "Développement web Fullstack (Frontend et Backend)",
        "Gestion de bases de données SQL et NoSQL",
        "Cloud computing sur AWS, GCP ou Azure",
        "Méthodologie DevOps et pipelines CI/CD",
        "Architecture logicielle et microservices"
    ],
    "Ingenierie_Technique": [
        "Gestion de projet technique et méthodologie Agile",
        "Modélisation mathématique et simulation",
        "Conception de systèmes embarqués et temps réel",
        "Cybersécurité et protection des systèmes d'information",
        "IoT et développement d'objets connectés",
        "Automatisation de processus et robotique"
    ]
}


# 2. PROFILS MÉTIERS ET PONDÉRATIONS (WEIGHTS)
# C'est ici que tu définis la logique "Métier".
# Clé = Nom du métier
# Valeur = Dictionnaire { "Nom_Du_Bloc": Poids }
# Ces poids sont utilisés dans la formule de la page 9 du PDF.

JOB_PROFILES = {
    # --- DROIT ---
    "Avocat / Juriste": {
        "Juridique_Reglementaire": 3, # Très important
        "Communication_Medias": 1     # Bonus : savoir s'exprimer
    },
    "Compliance Officer": {
        "Juridique_Reglementaire": 3,
        "Finance_Comptabilite": 1
    },

    # --- COMMERCE ---
    "Chef de Produit (Product Manager)": {
        "Marketing_Vente": 3,
        "Business_Strategie": 2,
        "Data_Analysis": 1 # De plus en plus demandé
    },
    "Business Developer": {
        "Business_Strategie": 3,
        "Marketing_Vente": 2
    },
    "Contrôleur de Gestion / Auditeur": {
        "Finance_Comptabilite": 3,
        "Data_Analysis": 1
    },
    "Responsable E-commerce": {
        "Marketing_Vente": 3,
        "Digital_Social": 2,
        "Business_Strategie": 1
    },

    # --- COM & DESIGN ---
    "Directeur Artistique / UX Designer": {
        "Creation_Design": 3,
        "Communication_Medias": 1
    },
    "Community Manager / Social Media Manager": {
        "Digital_Social": 3,
        "Communication_Medias": 2,
        "Creation_Design": 1
    },
    "Chargé de Communication": {
        "Communication_Medias": 3,
        "Digital_Social": 2
    },

    # --- INGÉNIEUR ---
    "Data Scientist": {
        "Machine_Learning_IA": 3,
        "Data_Analysis": 3,
        "Dev_Infrastructure": 1
    },
    "Data Analyst": {
        "Data_Analysis": 3,
        "Business_Strategie": 1, # Doit comprendre le business
        "Dev_Infrastructure": 1
    },
    "Développeur Full Stack": {
        "Dev_Infrastructure": 3,
        "Ingenierie_Technique": 1
    },
    "Ingénieur Cybersécurité": {
        "Ingenierie_Technique": 3,
        "Dev_Infrastructure": 2
    },
    "Architecte Cloud": {
        "Dev_Infrastructure": 3,
        "Ingenierie_Technique": 2
    }
}