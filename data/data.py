# data.py
# on stocke ici les blocs de compétences et les profils métiers en dur
# c'est un peu un doublon avec referentiel.json mais on s'en sert
# pour le scoring avec SBERT, chaque phrase sert de vecteur de comparaison

COMPETENCY_BLOCKS = {
    # les compétences juridiques, c'est surtout pour matcher le profil Juriste
    "Juridique_Reglementaire": [
        "Analyse et rédaction de contrats juridiques",
        "Veille juridique et réglementaire stricte",
        "Connaissance du droit des affaires et commercial",
        "Application du droit du travail et relations sociales",
        "Gestion des contentieux et procédures judiciaires",
        "Protection des données personnelles et RGPD"
    ],

    # compétences business classiques type école de commerce
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

    # compétences communication et créa
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

    # compétences techniques / ingénieur
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


# profils métiers avec les poids par bloc
# la clé c'est le nom du métier, la valeur c'est un dict bloc -> poids
# un poids de 3 = bloc principal, 2 = important, 1 = bonus
# on s'est basés sur les fiches ROME et un peu de bon sens pour les poids
JOB_PROFILES = {
    "Avocat / Juriste": {
        "Juridique_Reglementaire": 3,
        "Communication_Medias": 1     # faut savoir s'exprimer quand même
    },
    "Compliance Officer": {
        "Juridique_Reglementaire": 3,
        "Finance_Comptabilite": 1
    },
    "Chef de Produit (Product Manager)": {
        "Marketing_Vente": 3,
        "Business_Strategie": 2,
        "Data_Analysis": 1  # de plus en plus demandé en PM
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
    "Data Scientist": {
        "Machine_Learning_IA": 3,
        "Data_Analysis": 3,
        "Dev_Infrastructure": 1
    },
    "Data Analyst": {
        "Data_Analysis": 3,
        "Business_Strategie": 1,  
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
