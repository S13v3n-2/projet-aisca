# interface streamlit principale du projet AISCA
# on a injecte du CSS et JS custom pour avoir un rendu dark mode premium
# avec un step wizard au lieu du long formulaire a scroller
# c'est pas la facon la plus propre de faire mais pour un proto streamlit ca marche bien

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import json
from pathlib import Path

from scoring import get_models, load_and_index_data, analyze_profile, recommend_jobs, likert_to_semantic_text
from genai_augmentation import enrich_short_text, generate_learning_path, generate_professional_bio

st.set_page_config(
    page_title="AISCA - Orientation IA",
    page_icon="favicon.ico",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# on utilise Path pour remonter au bon dossier quel que soit le cwd
DATA_PATH = str(Path(__file__).parent.parent / "data" / "referentiel.json")

# on injecte tout le CSS custom pour le dark mode et la typo
# c'est un gros bloc mais c'est la seule facon de customiser streamlit a fond
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* reset du theme streamlit */
    .stApp {
        background-color: #0d1117 !important;
        color: #e6edf3 !important;
        font-family: 'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
    }

    /* cache la sidebar par defaut */
    [data-testid="stSidebar"] {
        display: none !important;
    }

    /* header streamlit */
    header[data-testid="stHeader"] {
        background-color: #0d1117 !important;
        border-bottom: 1px solid #21262d !important;
    }

    /* vire le padding top par defaut */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 1100px !important;
    }

    /* typo generale */
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        font-family: 'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
        color: #e6edf3 !important;
    }

    h1 {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }

    h2 {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
    }

    h3 {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }

    /* barre de progression custom */
    .progress-container {
        background: #161b22;
        border-radius: 8px;
        padding: 16px 24px;
        margin-bottom: 24px;
        border: 1px solid #21262d;
    }
    .progress-bar-bg {
        background: #21262d;
        border-radius: 4px;
        height: 6px;
        width: 100%;
        margin-top: 12px;
    }
    .progress-bar-fill {
        background: linear-gradient(90deg, #58a6ff, #79c0ff);
        border-radius: 4px;
        height: 6px;
        transition: width 0.4s ease;
    }
    .progress-steps {
        display: flex;
        justify-content: space-between;
        margin-top: 8px;
    }
    .progress-step {
        font-size: 0.7rem;
        color: #484f58;
        font-weight: 500;
    }
    .progress-step.active {
        color: #58a6ff;
    }
    .progress-step.done {
        color: #3fb950;
    }

    /* cards et surfaces */
    .card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
    }

    /* boutons navigation */
    .stButton > button {
        background: #21262d !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        padding: 8px 24px !important;
        font-weight: 500 !important;
        font-family: 'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: #30363d !important;
        border-color: #58a6ff !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #238636, #2ea043) !important;
        border-color: #238636 !important;
        color: white !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #2ea043, #3fb950) !important;
    }

    /* inputs et sliders */
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stTextArea textarea,
    .stTextInput input {
        background-color: #0d1117 !important;
        color: #e6edf3 !important;
        border-color: #30363d !important;
        border-radius: 8px !important;
    }
    .stTextArea textarea:focus,
    .stTextInput input:focus {
        border-color: #58a6ff !important;
        box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.15) !important;
    }

    /* metrics */
    [data-testid="stMetric"] {
        background: #161b22 !important;
        border: 1px solid #21262d !important;
        border-radius: 12px !important;
        padding: 16px !important;
    }
    [data-testid="stMetricValue"] {
        color: #58a6ff !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #8b949e !important;
    }

    /* tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #161b22 !important;
        border-radius: 8px !important;
        padding: 4px !important;
        gap: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #8b949e !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
    }
    .stTabs [aria-selected="true"] {
        background: #21262d !important;
        color: #e6edf3 !important;
    }

    /* divider */
    hr {
        border-color: #21262d !important;
    }

    /* progress bar streamlit */
    .stProgress > div > div {
        background: #21262d !important;
        border-radius: 4px !important;
    }
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #58a6ff, #79c0ff) !important;
        border-radius: 4px !important;
    }

    /* download button */
    .stDownloadButton > button {
        background: #161b22 !important;
        border-color: #30363d !important;
        color: #58a6ff !important;
    }

    /* balloons */
    .stBalloons { display: none !important; }

    /* step header */
    .step-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 20px;
    }
    .step-number {
        background: linear-gradient(135deg, #58a6ff, #388bfd);
        color: #0d1117;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.9rem;
        flex-shrink: 0;
    }
    .step-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #e6edf3;
    }
    .step-subtitle {
        font-size: 0.85rem;
        color: #8b949e;
        margin-top: 2px;
    }

    /* form submit override */
    [data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, #238636, #2ea043) !important;
        border-color: #238636 !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 10px 32px !important;
        border-radius: 8px !important;
    }

    /* hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* selectbox dropdown dark */
    [data-baseweb="select"] { background-color: #0d1117 !important; }
    [data-baseweb="popover"] { background-color: #161b22 !important; border-color: #30363d !important; }
    [data-baseweb="menu"] { background-color: #161b22 !important; }
    [role="option"] { background-color: #161b22 !important; }
    [role="option"]:hover { background-color: #21262d !important; }
</style>
""", unsafe_allow_html=True)


# icones SVG minimalistes a la place des emoji
ICONS = {
    "compass": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg>',
    "sliders": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>',
    "users": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
    "edit": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>',
    "wrench": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>',
    "briefcase": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="7" width="20" height="14" rx="2" ry="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/></svg>',
    "target": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
    "award": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#3fb950" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="7"/><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"/></svg>',
    "trending": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>',
    "book": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>',
    "user": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>',
}

STEP_NAMES = ["Domaines", "Niveaux", "Soft skills", "Experiences", "Outils", "Parcours"]


@st.cache_resource
def initialize_app():
    try:
        bi_model = get_models()
        data, comp_idx, comp_ids, embeddings = load_and_index_data(DATA_PATH)
        return bi_model, data, comp_idx, comp_ids, embeddings
    except Exception as e:
        st.error(f"Erreur d'initialisation : {e}")
        st.stop()


with st.spinner("Chargement du modele NLP..."):
    bi_model, data, comp_idx, comp_ids, embeddings = initialize_app()

# etat du wizard
if "current_step" not in st.session_state:
    st.session_state.current_step = 0
if "show_results" not in st.session_state:
    st.session_state.show_results = False
if "results_data" not in st.session_state:
    st.session_state.results_data = None


def render_progress_bar(current_step, total_steps, step_names):
    pct = int((current_step / total_steps) * 100)
    steps_html = ""
    for i, name in enumerate(step_names):
        css_class = "done" if i < current_step else "active" if i == current_step else ""
        steps_html += f'<span class="progress-step {css_class}">{name}</span>'
    st.markdown(f"""
    <div class="progress-container">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 0.85rem; font-weight: 600; color: #e6edf3;">Questionnaire</span>
            <span style="font-size: 0.8rem; color: #8b949e;">{current_step + 1} / {total_steps}</span>
        </div>
        <div class="progress-bar-bg">
            <div class="progress-bar-fill" style="width: {pct}%;"></div>
        </div>
        <div class="progress-steps">{steps_html}</div>
    </div>
    """, unsafe_allow_html=True)


def render_step_header(number, title, subtitle, icon_key="target"):
    icon = ICONS.get(icon_key, "")
    st.markdown(f"""
    <div class="step-header">
        <div class="step-number">{number}</div>
        <div>
            <div class="step-title">{icon} {title}</div>
            <div class="step-subtitle">{subtitle}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# radar chart avec Chart.js
def render_chartjs_radar(labels, values, title=""):
    labels_json = json.dumps(labels)
    values_json = json.dumps([round(v * 100, 1) for v in values])
    safe_id = title.replace(' ', '_').replace("'", "").replace('"', '')
    html = f"""
    <div style="background: #161b22; border-radius: 12px; padding: 20px; border: 1px solid #21262d;">
        <canvas id="radar_{safe_id}" width="400" height="350"></canvas>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        var ctx = document.getElementById('radar_{safe_id}').getContext('2d');
        new Chart(ctx, {{
            type: 'radar',
            data: {{
                labels: {labels_json},
                datasets: [{{
                    label: 'Votre profil',
                    data: {values_json},
                    backgroundColor: 'rgba(88, 166, 255, 0.15)',
                    borderColor: '#58a6ff',
                    borderWidth: 2,
                    pointBackgroundColor: '#58a6ff',
                    pointBorderColor: '#0d1117',
                    pointBorderWidth: 2,
                    pointRadius: 5,
                    pointHoverRadius: 7,
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: true,
                plugins: {{
                    legend: {{ display: false }},
                    title: {{
                        display: true,
                        text: '{title}',
                        color: '#e6edf3',
                        font: {{ size: 14, family: "'Inter', sans-serif", weight: '600' }}
                    }}
                }},
                scales: {{
                    r: {{
                        min: 0, max: 100,
                        ticks: {{ stepSize: 25, color: '#484f58', backdropColor: 'transparent', font: {{ size: 10 }} }},
                        grid: {{ color: '#21262d', lineWidth: 1 }},
                        angleLines: {{ color: '#21262d', lineWidth: 1 }},
                        pointLabels: {{ color: '#8b949e', font: {{ size: 11, family: "'Inter', sans-serif" }} }}
                    }}
                }}
            }}
        }});
    </script>
    """
    components.html(html, height=420)


# bar chart horizontal avec Chart.js
def render_chartjs_bar(labels, values, title=""):
    labels_json = json.dumps(labels)
    values_json = json.dumps([round(v * 100, 1) for v in values])
    colors = json.dumps(['#58a6ff' if v >= 0.7 else '#d29922' if v >= 0.4 else '#f85149' for v in values])
    safe_id = title.replace(' ', '_').replace("'", "").replace('"', '')
    html = f"""
    <div style="background: #161b22; border-radius: 12px; padding: 20px; border: 1px solid #21262d;">
        <canvas id="bar_{safe_id}" width="400" height="250"></canvas>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        var ctx = document.getElementById('bar_{safe_id}').getContext('2d');
        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: {labels_json},
                datasets: [{{ data: {values_json}, backgroundColor: {colors}, borderRadius: 6, barThickness: 28 }}]
            }},
            options: {{
                responsive: true, indexAxis: 'y',
                plugins: {{
                    legend: {{ display: false }},
                    title: {{ display: true, text: '{title}', color: '#e6edf3', font: {{ size: 14, family: "'Inter', sans-serif", weight: '600' }} }}
                }},
                scales: {{
                    x: {{ min: 0, max: 100, ticks: {{ color: '#484f58', callback: function(v) {{ return v + '%'; }} }}, grid: {{ color: '#21262d' }} }},
                    y: {{ ticks: {{ color: '#8b949e', font: {{ size: 11 }} }}, grid: {{ display: false }} }}
                }}
            }}
        }});
    </script>
    """
    components.html(html, height=320)


# profils de test pre-definis
TEST_PROFILES = {
    "-- Aucun (remplir manuellement) --": None,
    "Data Scientist": {
        "business": "Débutant", "finance": "Débutant", "design": "Débutant",
        "communication": "Débutant", "data_analysis": "Expert", "ml": "Expert",
        "dev": "Avancé", "engineering": "Débutant",
        "rigueur": "Rigoureux et attentif aux détails",
        "leadership": "Je préfère contribuer en tant que membre",
        "persuasion": "Parfois, selon le sujet", "empathie": "Je trouve un équilibre",
        "projet": "J'ai développé un système de prédiction de churn client avec Python, scikit-learn et XGBoost. Modèles de classification supervisée à 92% de précision, déployés via FastAPI et Docker avec pipelines MLOps.",
        "journee": "Analyser des datasets complexes, entraîner des modèles de machine learning, évaluer leurs performances et déployer des pipelines de données en production.",
        "interet": "Machine Learning, Deep Learning, NLP, transformers, modèles prédictifs, TensorFlow, PyTorch, HuggingFace.",
        "defis": "Optimiser des algorithmes de machine learning, améliorer la précision des modèles, trouver des patterns dans des datasets massifs.",
        "objectif": "Devenir Lead Data Scientist ou ML Engineer spécialisé en NLP et déploiement de modèles IA en production.",
        "outils_data": ["Python", "Pandas", "SQL", "Matplotlib", "NumPy", "Seaborn"],
        "outils_ml": ["Scikit-learn", "TensorFlow", "PyTorch", "HuggingFace", "BERT"],
        "outils_dev": ["Git", "Docker", "FastAPI"],
        "outils_design": [], "outils_marketing": [],
        "formation": "Data Science et Machine Learning", "etudes": "Bac+5 (Master)",
        "experience": "3-5 ans", "secteur": "Intelligence Artificielle et Data"
    },
    "Marketing / Communication": {
        "business": "Intermédiaire", "finance": "Débutant", "design": "Intermédiaire",
        "communication": "Expert", "data_analysis": "Débutant", "ml": "Débutant",
        "dev": "Débutant", "engineering": "Débutant",
        "rigueur": "Créatif et improvisateur", "leadership": "Je prends naturellement le lead",
        "persuasion": "Oui, j'adore argumenter et persuader", "empathie": "Je suis très empathique et à l'écoute",
        "projet": "J'ai piloté la stratégie de communication d'une startup : refonte de l'identité de marque, gestion des réseaux sociaux Instagram et LinkedIn, campagnes publicitaires Meta Ads avec +40% d'engagement.",
        "journee": "Créer du contenu engageant pour les réseaux sociaux, rédiger des communiqués de presse, analyser les performances des campagnes.",
        "interet": "Stratégie de marque, storytelling, réseaux sociaux, relations presse, influence marketing, SEO SEA.",
        "defis": "Construire une image de marque forte, toucher des audiences ciblées et mesurer l'impact des campagnes.",
        "objectif": "Devenir Directrice de la Communication ou Social Media Manager dans une agence créative internationale.",
        "outils_data": ["Excel avancé"], "outils_ml": [], "outils_dev": [],
        "outils_design": ["Canva", "InDesign", "Photoshop"],
        "outils_marketing": ["Meta Business Suite", "Google Analytics", "Hootsuite", "Mailchimp", "HubSpot"],
        "formation": "Communication et Marketing Digital", "etudes": "Bac+5 (Master)",
        "experience": "3-5 ans", "secteur": "Agence de communication et marketing"
    },
    "Designer / Creatif": {
        "business": "Débutant", "finance": "Débutant", "design": "Expert",
        "communication": "Avancé", "data_analysis": "Débutant", "ml": "Débutant",
        "dev": "Intermédiaire", "engineering": "Débutant",
        "rigueur": "Créatif et improvisateur", "leadership": "Cela dépend du contexte",
        "persuasion": "Parfois, selon le sujet", "empathie": "Je suis très empathique et à l'écoute",
        "projet": "J'ai conçu l'identité visuelle complète d'une marque de cosmétiques : logo, charte graphique, packaging, site web et supports print.",
        "journee": "Concevoir des visuels créatifs, travailler sur des mises en page print et digital.",
        "interet": "Design graphique, direction artistique, typographie, identité visuelle, motion design, UI/UX.",
        "defis": "Créer des identités visuelles mémorables, innover dans les tendances design.",
        "objectif": "Devenir Directeur Artistique ou UX/UI Designer Lead dans une agence de design internationale.",
        "outils_data": [], "outils_ml": [], "outils_dev": ["Git"],
        "outils_design": ["Photoshop", "Illustrator", "InDesign", "Figma", "After Effects"],
        "outils_marketing": ["Google Analytics"],
        "formation": "Design Graphique et Direction Artistique", "etudes": "Bac+5 (Master)",
        "experience": "3-5 ans", "secteur": "Agence créative et design"
    },
    "Juriste / Consultant": {
        "business": "Intermédiaire", "finance": "Intermédiaire", "design": "Débutant",
        "communication": "Avancé", "data_analysis": "Débutant", "ml": "Débutant",
        "dev": "Débutant", "engineering": "Débutant", "_juridique_level": "Expert",
        "rigueur": "Rigoureux et attentif aux détails", "leadership": "Je prends naturellement le lead",
        "persuasion": "Oui, j'adore argumenter et persuader", "empathie": "Je trouve un équilibre",
        "projet": "J'ai accompagné une PME dans sa mise en conformité RGPD : audit des traitements de données personnelles, rédaction des contrats DPA et des actes juridiques.",
        "journee": "Analyser des contrats et textes juridiques complexes, rédiger des actes juridiques, conseiller sur les risques réglementaires.",
        "interet": "Droit des affaires, droit des contrats, conformité RGPD, propriété intellectuelle, droit du travail.",
        "defis": "Sécuriser juridiquement les opérations d'entreprise, interpréter des textes juridiques complexes.",
        "objectif": "Devenir Juriste d'entreprise Senior ou Avocat d'affaires spécialisé en droit du numérique.",
        "outils_data": ["Excel avancé"], "outils_ml": [], "outils_dev": [],
        "outils_design": [], "outils_marketing": [],
        "formation": "Droit des affaires et Conformité réglementaire", "etudes": "Bac+5 (Master)",
        "experience": "3-5 ans", "secteur": "Cabinet d'avocats et conseil juridique"
    },
}

# header
st.markdown(f"""
<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
    {ICONS['target']}
    <h1 style="margin: 0; padding: 0;">AISCA</h1>
</div>
<p style="color: #8b949e; font-size: 0.9rem; margin-top: 0; margin-bottom: 24px;">
    Agent Intelligent de Cartographie des Competences
</p>
""", unsafe_allow_html=True)

# mode resultats
if st.session_state.show_results and st.session_state.results_data:
    rd = st.session_state.results_data
    top_jobs = rd["top_jobs"]
    user_text = rd["user_text"]
    comp_scores = rd["comp_scores"]

    if st.button("Nouveau questionnaire"):
        st.session_state.show_results = False
        st.session_state.results_data = None
        st.session_state.current_step = 0
        st.rerun()

    st.markdown("---")

    st.markdown(f'<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">{ICONS["award"]}<h2 style="margin: 0;">Resultats de l\'analyse</h2></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    overall_score = top_jobs[0]['score'] if top_jobs else 0
    with col1:
        st.metric("Score de match #1", f"{overall_score:.0%}")
    with col2:
        nb_fortes = sum(1 for b in top_jobs[0]['blocs'] if b['score'] >= 0.7)
        st.metric("Blocs maitrises", f"{nb_fortes}/{len(top_jobs[0]['blocs'])}")
    with col3:
        potentiel = "Excellent" if overall_score >= 0.75 else "Tres bon" if overall_score >= 0.6 else "Bon"
        st.metric("Potentiel", potentiel)

    st.markdown("---")
    st.markdown(f'<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 16px;">{ICONS["trending"]}<h2 style="margin: 0;">Top 3 des metiers recommandes</h2></div>', unsafe_allow_html=True)

    tabs = st.tabs([f"#{i+1}  {job['titre']}  ({job['score']:.0%})" for i, job in enumerate(top_jobs)])

    for i, (tab, job) in enumerate(zip(tabs, top_jobs)):
        with tab:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"### {job['titre']}")
                global_score = float(job['score'])
                st.progress(max(0.0, min(1.0, global_score)))
                if job.get('description'):
                    st.markdown(f'<div class="card"><span style="color: #8b949e; font-size: 0.85rem;">{job["description"]}</span></div>', unsafe_allow_html=True)
                if job.get('blocs') and len(job['blocs']) > 0:
                    labels = [b['nom'] for b in job['blocs']]
                    values = [float(b['score']) for b in job['blocs']]
                    if len(labels) >= 3:
                        render_chartjs_radar(labels, values, f"Competences - {job['titre']}")
                    else:
                        render_chartjs_bar(labels, values, f"Adequation - {job['titre']}")
            with col2:
                st.metric("Score global", f"{global_score:.0%}")
                if i < len(top_jobs) - 1:
                    delta = global_score - float(top_jobs[i+1]['score'])
                    st.metric(f"Ecart avec #{i+2}", f"+{delta:.1%}")
                st.markdown("---")
                with st.expander("Detail des blocs", expanded=True):
                    for bloc in job['blocs']:
                        b_score = float(bloc['score'])
                        st.write(f"**{bloc['nom']}**")
                        st.progress(max(0.0, min(1.0, b_score)))
                        st.caption(f"{b_score:.0%}")

    st.markdown("---")
    st.markdown(f'<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 16px;">{ICONS["book"]}<h2 style="margin: 0;">Recommandations</h2></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("#### Plan de progression")
        weak_blocs = {b['nom']: b['score'] for b in top_jobs[0]['blocs'] if b['score'] < 0.6}
        if weak_blocs:
            with st.spinner("Generation du plan..."):
                learning_path = generate_learning_path(weak_blocs, top_jobs[0]['titre'], user_text[:600])
            st.markdown(learning_path)
        else:
            st.success("Vous maitrisez deja tous les blocs requis.")
    with col2:
        st.markdown("#### Bio professionnelle")
        strong_blocs = {b['nom']: b['score'] for b in top_jobs[0]['blocs'] if b['score'] >= 0.6}
        with st.spinner("Redaction..."):
            bio = generate_professional_bio(
                strong_blocs, top_jobs[0]['titre'],
                {"projet_tech": rd.get("projet_tech", ""), "objectif": rd.get("objectif", "")}
            )
        st.info(bio)
        st.download_button("Telecharger ma bio", bio, file_name="bio_aisca.txt", mime="text/plain", width="stretch")

    st.markdown("---")
    df_scores = pd.DataFrame([
        {"Bloc": b['nom'], "Score": f"{b['score']:.0%}",
         "Niveau": "Maitrise" if b['score'] >= 0.7 else "Intermediaire" if b['score'] >= 0.4 else "A developper"}
        for b in top_jobs[0]['blocs']
    ]).sort_values("Score", ascending=False)
    st.dataframe(df_scores, width="stretch", hide_index=True)

    with st.expander("DEBUG - Details techniques", expanded=False):
        st.write(f"**Mots :** {len(user_text.split())} | **Caracteres :** {len(user_text)}")
        st.text_area("Texte envoye au modele :", user_text, height=150)
        if comp_scores:
            top_comps = sorted(comp_scores.items(), key=lambda x: -x[1])[:10]
            st.write("**Top 10 competences :**")
            for cid, score in top_comps:
                if cid in comp_idx:
                    st.write(f"- {comp_idx[cid]['texte']} : {score:.2%}")

else:
    # mode questionnaire avec step wizard

    # dropdown profil de test
    selected_profile = st.selectbox(
        "Profil de test (optionnel)",
        list(TEST_PROFILES.keys()), index=0,
        help="Selectionner un profil pre-rempli pour la demo"
    )
    if selected_profile != "-- Aucun (remplir manuellement) --":
        profile = TEST_PROFILES[selected_profile]
        if profile and st.session_state.get("_last_profile") != selected_profile:
            for k, v in profile.items():
                st.session_state[k] = v
            st.session_state["_last_profile"] = selected_profile
            st.rerun()

    current = st.session_state.current_step
    render_progress_bar(current, len(STEP_NAMES), STEP_NAMES)

    with st.form(key="wizard_form"):

        if current == 0:
            render_step_header(1, "Domaines d'interet", "Selectionnez les domaines qui vous attirent", "compass")
            col1, col2, col3 = st.columns(3)
            with col1:
                domaines_col1 = st.multiselect("Business & juridique",
                    ["Juridique", "Business et Stratégie", "Marketing et Vente", "Finance et Comptabilité"])
            with col2:
                domaines_col2 = st.multiselect("Creatif & communication",
                    ["Communication et Médias", "Création et Design", "Digital et Réseaux Sociaux"])
            with col3:
                domaines_col3 = st.multiselect("Technique & data",
                    ["Data Analysis", "Machine Learning et IA", "Développement et Infrastructure", "Ingénierie et Technique"])

        elif current == 1:
            render_step_header(2, "Auto-evaluation technique", "Evaluez votre niveau dans chaque domaine", "sliders")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Business & Strategie**")
                business_level = st.select_slider("Strategie d'entreprise",
                    options=["Débutant", "Intermédiaire", "Avancé", "Expert"],
                    value=st.session_state.get("business", "Débutant"), key="business")
                finance_level = st.select_slider("Finance / Comptabilite",
                    options=["Débutant", "Intermédiaire", "Avancé", "Expert"],
                    value=st.session_state.get("finance", "Débutant"), key="finance")
                st.markdown("**Data & Analyse**")
                data_analysis_level = st.select_slider("Analyse de donnees",
                    options=["Débutant", "Intermédiaire", "Avancé", "Expert"],
                    value=st.session_state.get("data_analysis", "Débutant"), key="data_analysis")
                ml_level = st.select_slider("Machine Learning / IA",
                    options=["Débutant", "Intermédiaire", "Avancé", "Expert"],
                    value=st.session_state.get("ml", "Débutant"), key="ml")
            with col2:
                st.markdown("**Creativite & Communication**")
                design_level = st.select_slider("Design / Creativite",
                    options=["Débutant", "Intermédiaire", "Avancé", "Expert"],
                    value=st.session_state.get("design", "Débutant"), key="design")
                communication_level = st.select_slider("Communication / Media",
                    options=["Débutant", "Intermédiaire", "Avancé", "Expert"],
                    value=st.session_state.get("communication", "Débutant"), key="communication")
                st.markdown("**Developpement & Tech**")
                dev_level = st.select_slider("Developpement logiciel",
                    options=["Débutant", "Intermédiaire", "Avancé", "Expert"],
                    value=st.session_state.get("dev", "Débutant"), key="dev")
                engineering_level = st.select_slider("Ingenierie technique",
                    options=["Débutant", "Intermédiaire", "Avancé", "Expert"],
                    value=st.session_state.get("engineering", "Débutant"), key="engineering")

        elif current == 2:
            render_step_header(3, "Soft skills", "Vos traits de personnalite", "users")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Style de travail**")
                rigoeur = st.radio("Vous etes plutot :",
                    ["Rigoureux et attentif aux détails", "Créatif et improvisateur", "Équilibré entre les deux"], key="rigueur")
                leadership = st.radio("Face a un projet d'equipe :",
                    ["Je prends naturellement le lead", "Je préfère contribuer en tant que membre", "Cela dépend du contexte"], key="leadership")
            with col2:
                st.markdown("**Competences relationnelles**")
                persuasion = st.radio("Debattre et convaincre ?",
                    ["Oui, j'adore argumenter et persuader", "Non, je préfère éviter les confrontations", "Parfois, selon le sujet"], key="persuasion")
                empathie = st.radio("Relations professionnelles :",
                    ["Je suis très empathique et à l'écoute", "Je privilégie l'efficacité sur l'émotion", "Je trouve un équilibre"], key="empathie")

        elif current == 3:
            render_step_header(4, "Experiences et aspirations", "Detaillez pour une analyse plus precise", "edit")
            projet_tech = st.text_area("Un projet dont vous etes fier",
                placeholder="Ex : J'ai developpe un dashboard Power BI...", height=100, key="projet")
            col1, col2 = st.columns(2)
            with col1:
                journee_ideale = st.text_area("Journee de travail ideale",
                    placeholder="Ex : Alterner entre analyse de donnees...", height=90, key="journee")
                interet_specifique = st.text_area("Domaines qui vous passionnent",
                    placeholder="Ex : L'ethique de l'IA...", height=90, key="interet")
            with col2:
                defis_aimes = st.text_area("Defis qui vous stimulent",
                    placeholder="Ex : Resoudre des problemes complexes...", height=90, key="defis")
                objectif_carriere = st.text_area("Objectif de carriere a 3-5 ans",
                    placeholder="Ex : Devenir Lead Data Scientist...", height=90, key="objectif")

        elif current == 4:
            render_step_header(5, "Outils et technologies", "Selectionnez ce que vous maitrisez", "wrench")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Data & Analytics**")
                outils_data = st.multiselect("Outils data",
                    ["Python", "R", "SQL", "Excel avancé", "Pandas", "NumPy", "Matplotlib", "Seaborn", "Plotly", "Tableau", "Power BI"], key="outils_data")
            with col2:
                st.markdown("**IA & Machine Learning**")
                outils_ml = st.multiselect("Frameworks IA",
                    ["Scikit-learn", "TensorFlow", "PyTorch", "Keras", "HuggingFace", "LangChain", "OpenAI API", "BERT", "GPT"], key="outils_ml")
            with col3:
                st.markdown("**Dev & Cloud**")
                outils_dev = st.multiselect("Stack technique",
                    ["Git", "Docker", "Kubernetes", "AWS", "Azure", "GCP", "Flask", "Django", "FastAPI", "React", "Node.js", "MongoDB"], key="outils_dev")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Design & Creativite**")
                outils_design = st.multiselect("Suite creative",
                    ["Photoshop", "Illustrator", "InDesign", "Figma", "Canva", "After Effects", "Premiere Pro", "Sketch"], key="outils_design")
            with col2:
                st.markdown("**Marketing & Communication**")
                outils_marketing = st.multiselect("Outils marketing",
                    ["Google Analytics", "Meta Business Suite", "Google Ads", "SEMrush", "Mailchimp", "HubSpot", "Hootsuite", "Buffer"], key="outils_marketing")

        elif current == 5:
            render_step_header(6, "Parcours et experience", "Formation et experience professionnelle", "briefcase")
            col1, col2 = st.columns(2)
            with col1:
                niveau_etudes = st.selectbox("Niveau d'etudes",
                    ["Bac", "Bac+2 (BTS/DUT)", "Bac+3 (Licence)", "Bac+5 (Master)", "Bac+8 (Doctorat)", "Autre"], key="etudes")
                domaine_formation = st.text_input("Domaine de formation",
                    placeholder="Ex : Data Science, Marketing...", key="formation")
            with col2:
                annees_experience = st.selectbox("Experience professionnelle",
                    ["Étudiant / 0 an", "0-2 ans", "3-5 ans", "6-10 ans", "+10 ans"], key="experience")
                secteur_actuel = st.text_input("Secteur d'activite",
                    placeholder="Ex : Banque, E-commerce...", key="secteur")

        # bouton submit visible seulement a la derniere etape
        if current == 5:
            submit = st.form_submit_button("Analyser mon profil", type="primary", use_container_width=True)
        else:
            submit = st.form_submit_button("_", disabled=True)
            st.markdown('<style>button[data-testid="stFormSubmitButton"][disabled]{display:none !important;}</style>', unsafe_allow_html=True)

    # navigation
    col_prev, col_spacer, col_next = st.columns([1, 2, 1])
    with col_prev:
        if current > 0:
            if st.button("Precedent", use_container_width=True):
                st.session_state.current_step -= 1
                st.rerun()
    with col_next:
        if current < 5:
            if st.button("Suivant", use_container_width=True, type="primary"):
                st.session_state.current_step += 1
                st.rerun()

    # traitement a la soumission
    if current == 5 and submit:
        likert_levels = {
            "business": st.session_state.get("business", "Débutant"),
            "finance": st.session_state.get("finance", "Débutant"),
            "design": st.session_state.get("design", "Débutant"),
            "communication": st.session_state.get("communication", "Débutant"),
            "data_analysis": st.session_state.get("data_analysis", "Débutant"),
            "ml": st.session_state.get("ml", "Débutant"),
            "dev": st.session_state.get("dev", "Débutant"),
            "engineering": st.session_state.get("engineering", "Débutant"),
        }
        juridique_level = st.session_state.get("_juridique_level", None)
        if juridique_level:
            likert_levels["juridique"] = juridique_level

        likert_summary = likert_to_semantic_text(likert_levels)

        soft_skills_summary = f"Style de travail : {st.session_state.get('rigueur', '')}. Leadership : {st.session_state.get('leadership', '')}. Persuasion : {st.session_state.get('persuasion', '')}. Empathie : {st.session_state.get('empathie', '')}."

        outils_all = (st.session_state.get("outils_data", []) + st.session_state.get("outils_ml", []) +
                      st.session_state.get("outils_dev", []) + st.session_state.get("outils_design", []) +
                      st.session_state.get("outils_marketing", []))

        _projet = st.session_state.get("projet", "")
        _interet = st.session_state.get("interet", "")
        _objectif = st.session_state.get("objectif", "")

        background_summary = f"Formation : {st.session_state.get('etudes', '')} en {st.session_state.get('formation', '')}. Experience : {st.session_state.get('experience', '')} dans le secteur {st.session_state.get('secteur', '')}."
        outils_text = f"Je maîtrise : {', '.join(outils_all)}." if outils_all else ""

        juridique_boost = ""
        if st.session_state.get("_juridique_level") in ("Expert", "Avancé"):
            repetitions = 5 if st.session_state.get("_juridique_level") == "Expert" else 3
            juridique_raw = (
                "Analyser et interpréter des contrats et textes juridiques complexes. "
                "Maîtriser le droit des affaires et le droit commercial. "
                "Rédiger des actes juridiques et documents légaux professionnels. "
                "Assurer la conformité réglementaire et le respect des normes RGPD. "
                "Maîtriser le droit du travail et les relations sociales en entreprise. "
                "Effectuer une veille réglementaire et juridique continue. "
                "Gérer les procédures juridiques et le contentieux. "
                "Protéger la propriété intellectuelle et les brevets. "
                "Aimer débattre et argumenter, être rigoureux dans les détails, "
                "analyser des problèmes juridiques complexes."
            )
            juridique_boost = " ".join([juridique_raw] * repetitions)

        user_text = " ".join([
            juridique_boost, likert_summary, soft_skills_summary,
            _projet, st.session_state.get("journee", ""), _interet,
            st.session_state.get("defis", ""), _objectif,
            outils_text, background_summary
        ]).strip()

        if len(user_text.split()) < 30:
            st.warning("Reponses trop courtes. Detaillez davantage vos experiences.")
        else:
            with st.spinner("Pretraitement semantique..."):
                enriched_projet = enrich_short_text(_projet) if len(_projet.split()) < 15 else _projet
                enriched_interet = enrich_short_text(_interet) if len(_interet.split()) < 10 else _interet
                final_text = user_text.replace(_projet, enriched_projet).replace(_interet, enriched_interet)

            with st.spinner("Analyse NLP en cours..."):
                comp_scores = analyze_profile(final_text, bi_model, comp_ids, embeddings, None, comp_idx, top_k=None)
                top_jobs, bloc_scores = recommend_jobs(comp_scores, data, top_n=3)

            st.session_state.results_data = {
                "top_jobs": top_jobs, "bloc_scores": bloc_scores,
                "user_text": user_text, "final_text": final_text,
                "comp_scores": comp_scores, "projet_tech": _projet, "objectif": _objectif,
            }
            st.session_state.show_results = True
            st.rerun()

# footer
st.markdown("---")
st.markdown('<p style="color: #484f58; font-size: 0.75rem; text-align: center;">AISCA - 2026 | EFREI - Master Data Engineering & AI | Projet IA Generative</p>', unsafe_allow_html=True)
