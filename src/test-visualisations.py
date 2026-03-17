# interface streamlit principale du projet AISCA
# on a injecte du CSS et JS custom pour avoir un rendu dark mode premium
# avec un step wizard au lieu du long formulaire a scroller
# fusion de sauvegarde-front (CSS responsive, compétences par bloc, radar responsive)
# et test-visualisations.py (results_data enrichi, generate_learning_path avec profil complet)

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import json
from pathlib import Path

from scoring import (
    get_models, load_and_index_data, analyze_profile, recommend_jobs,
    likert_to_semantic_text, build_profile_embedding, get_domaine_embeddings
)
from genai_augmentation import enrich_short_text, generate_learning_path, generate_professional_bio

st.set_page_config(
    page_title="AISCA",
    page_icon="favicon.ico",
    layout="wide",
    initial_sidebar_state="collapsed"
)

DATA_PATH = str(Path(__file__).parent.parent / "data" / "referentiel.json")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        background-color: #0d1117 !important;
        color: #e6edf3 !important;
        font-family: 'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
    }

    [data-testid="stSidebar"] { display: none !important; }

    header[data-testid="stHeader"] {
        background-color: #0d1117 !important;
        border-bottom: 1px solid #21262d !important;
    }

    /* =====================================================
       RESPONSIVE — PLEIN ÉCRAN
       Streamlit wide mode empile plusieurs conteneurs avec
       max-width et margin:auto. On doit tous les neutraliser.
       ===================================================== */
    .block-container,
    [data-testid="stMainBlockContainer"],
    [data-testid="stBlockContainer"],
    [data-testid="stAppViewBlockContainer"],
    section[data-testid="stMain"] > div,
    .main > div {
        max-width: 100% !important;
        width: 100% !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
        box-sizing: border-box !important;
    }
    .block-container,
    [data-testid="stMainBlockContainer"] {
        padding-top: 4rem !important;
        padding-bottom: 2rem !important;
    }
    @media (min-width: 1800px) {
        .block-container, [data-testid="stMainBlockContainer"] {
            padding-left: 6rem !important; padding-right: 6rem !important;
        }
    }
    @media (min-width: 1400px) and (max-width: 1799px) {
        .block-container, [data-testid="stMainBlockContainer"] {
            padding-left: 4rem !important; padding-right: 4rem !important;
        }
    }
    @media (max-width: 900px) {
        .block-container, [data-testid="stMainBlockContainer"] {
            padding-left: 1rem !important; padding-right: 1rem !important;
        }
        [data-testid="column"] {
            min-width: 100% !important; flex: 1 1 100% !important;
        }
    }

    /* typo generale — IMPORTANT : on n'inclut PAS span dans ce reset.
       Depuis Streamlit 1.47, les icones des expanders utilisent Material Symbols :
       ligatures dans un span. Forcer font-family sur tous les spans casse ces
       ligatures et affiche "keyboard_arrow_down" en texte brut. */
    h1, h2, h3, h4, h5, h6, p, label, div,
    button, input, textarea, select {
        font-family: 'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
        color: #e6edf3 !important;
    }

    h1 { font-size: 1.8rem !important; font-weight: 700 !important; letter-spacing: -0.02em !important; }
    h2 { font-size: 1.3rem !important; font-weight: 600 !important; letter-spacing: -0.01em !important; }
    h3 { font-size: 1.1rem !important; font-weight: 600 !important; }

    .progress-container {
        background: #161b22; border-radius: 8px; padding: 16px 24px;
        margin-bottom: 24px; border: 1px solid #21262d;
    }
    .progress-bar-bg { background: #21262d; border-radius: 4px; height: 6px; width: 100%; margin-top: 12px; }
    .progress-bar-fill {
        background: linear-gradient(90deg, #58a6ff, #79c0ff);
        border-radius: 4px; height: 6px; transition: width 0.4s ease;
    }
    .progress-steps { display: flex; justify-content: space-between; margin-top: 8px; }
    .progress-step { font-size: 0.7rem; color: #484f58; font-weight: 500; }
    .progress-step.active { color: #58a6ff; }
    .progress-step.done { color: #3fb950; }

    .card { background: #161b22; border: 1px solid #21262d; border-radius: 12px; padding: 24px; margin-bottom: 16px; }

    details[data-testid="stExpander"] {
        background: #0d1117 !important; border: 1px solid #21262d !important;
        border-radius: 8px !important; margin-bottom: 6px !important;
    }
    details[data-testid="stExpander"] summary {
        padding: 10px 14px !important; font-weight: 500 !important; font-size: 0.9rem !important;
    }
    details[data-testid="stExpander"][open] { border-color: #30363d !important; }

    .stButton > button {
        background: #21262d !important; color: #e6edf3 !important;
        border: 1px solid #30363d !important; border-radius: 8px !important;
        padding: 8px 24px !important; font-weight: 500 !important;
        font-family: 'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover { background: #30363d !important; border-color: #58a6ff !important; }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #238636, #2ea043) !important;
        border-color: #238636 !important; color: white !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #2ea043, #3fb950) !important;
    }

    .stSelectbox > div > div, .stMultiSelect > div > div,
    .stTextArea textarea, .stTextInput input {
        background-color: #0d1117 !important; color: #e6edf3 !important;
        border-color: #30363d !important; border-radius: 8px !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #58a6ff !important; box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.15) !important;
    }

    [data-testid="stMetric"] {
        background: #161b22 !important; border: 1px solid #21262d !important;
        border-radius: 12px !important; padding: 16px !important;
    }
    [data-testid="stMetricValue"] { color: #58a6ff !important; font-weight: 700 !important; }
    [data-testid="stMetricLabel"] { color: #8b949e !important; }

    .stTabs [data-baseweb="tab-list"] {
        background: #161b22 !important; border-radius: 8px !important; padding: 4px !important; gap: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important; color: #8b949e !important;
        border-radius: 6px !important; font-weight: 500 !important;
    }
    .stTabs [aria-selected="true"] { background: #21262d !important; color: #e6edf3 !important; }

    hr { border-color: #21262d !important; }

    .stProgress > div > div { background: #21262d !important; border-radius: 4px !important; }
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #58a6ff, #79c0ff) !important; border-radius: 4px !important;
    }

    .stDownloadButton > button {
        background: #161b22 !important; border-color: #30363d !important; color: #58a6ff !important;
    }

    .stBalloons { display: none !important; }

    .step-header { display: flex; align-items: center; gap: 12px; margin-bottom: 20px; }
    .step-number {
        background: linear-gradient(135deg, #58a6ff, #388bfd); color: #0d1117;
        width: 32px; height: 32px; border-radius: 50%; display: flex;
        align-items: center; justify-content: center; font-weight: 700; font-size: 0.9rem; flex-shrink: 0;
    }
    .step-title { font-size: 1.2rem; font-weight: 600; color: #e6edf3; }
    .step-subtitle { font-size: 0.85rem; color: #8b949e; margin-top: 2px; }

    #MainMenu {visibility: hidden;} footer {visibility: hidden;} .stDeployButton {display: none;}

    [data-baseweb="select"] { background-color: #0d1117 !important; }
    [data-baseweb="popover"] { background-color: #161b22 !important; border-color: #30363d !important; }
    [data-baseweb="menu"] { background-color: #161b22 !important; }
    [role="option"] { background-color: #161b22 !important; }
    [role="option"]:hover { background-color: #21262d !important; }
</style>
""", unsafe_allow_html=True)


ICONS = {
    "compass":   '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg>',
    "sliders":   '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>',
    "users":     '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
    "edit":      '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>',
    "wrench":    '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>',
    "briefcase": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="7" width="20" height="14" rx="2" ry="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/></svg>',
    "target":    '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
    "award":     '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#3fb950" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="7"/><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"/></svg>',
    "trending":  '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>',
    "book":      '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>',
    "user":      '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>',
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

if "current_step" not in st.session_state:
    st.session_state.current_step = 0
if "show_results" not in st.session_state:
    st.session_state.show_results = False
if "results_data" not in st.session_state:
    st.session_state.results_data = None


def render_progress_bar(current_step, total_steps, step_names):
    # barre alignée exactement sous le label de l'étape courante
    # étape 0 → 0%, étape 5 → 100%
    pct = int((current_step / (total_steps - 1)) * 100) if total_steps > 1 else 0
    steps_html = ""
    for i, name in enumerate(step_names):
        css_class = "done" if i < current_step else "active" if i == current_step else ""
        steps_html += f'<span class="progress-step {css_class}">{name}</span>'
    st.markdown(f"""
    <div class="progress-container">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:0.85rem; font-weight:600; color:#e6edf3;">Questionnaire</span>
            <span style="font-size:0.8rem; color:#8b949e;">{current_step + 1} / {total_steps}</span>
        </div>
        <div class="progress-bar-bg">
            <div class="progress-bar-fill" style="width:{pct}%;"></div>
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


# radar chart responsive Chart.js — maintainAspectRatio false pour s'adapter au conteneur
def render_chartjs_radar(labels, values, title=""):
    labels_json = json.dumps(labels)
    values_json = json.dumps([round(v * 100, 1) for v in values])
    target_json = json.dumps([70] * len(labels))
    safe_id = title.replace(' ', '_').replace("'", "").replace('"', '').replace('—', '').replace(' ', '_')
    html = f"""
    <div style="background:#161b22; border-radius:12px; padding:20px; border:1px solid #21262d; width:100%; box-sizing:border-box;">
        <div style="position:relative; width:100%; height:420px;">
            <canvas id="radar_{safe_id}"></canvas>
        </div>
        <div style="display:flex;gap:20px;justify-content:center;margin-top:12px;font-size:12px;color:#8b949e;">
            <span style="display:flex;align-items:center;gap:6px;">
                <span style="width:14px;height:3px;background:#58a6ff;border-radius:2px;display:inline-block;"></span>Votre profil
            </span>
            <span style="display:flex;align-items:center;gap:6px;">
                <span style="width:14px;height:2px;background:#3fb950;border-radius:2px;display:inline-block;border-top:2px dashed #3fb950;"></span>Seuil maîtrise (70%)
            </span>
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script>
    (function() {{
        var canvas = document.getElementById('radar_{safe_id}');
        new Chart(canvas, {{
            type: 'radar',
            data: {{
                labels: {labels_json},
                datasets: [
                    {{
                        label: 'Votre profil',
                        data: {values_json},
                        backgroundColor: 'rgba(88, 166, 255, 0.15)',
                        borderColor: '#58a6ff', borderWidth: 2.5,
                        pointBackgroundColor: '#58a6ff', pointBorderColor: '#0d1117',
                        pointBorderWidth: 2, pointRadius: 5, pointHoverRadius: 7,
                    }},
                    {{
                        label: 'Seuil maitrise',
                        data: {target_json},
                        backgroundColor: 'rgba(63, 185, 80, 0.05)',
                        borderColor: '#3fb950', borderWidth: 1.5,
                        borderDash: [6, 4], pointRadius: 0, pointHoverRadius: 0,
                    }}
                ]
            }},
            options: {{
                responsive: true, maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }},
                    title: {{
                        display: true, text: '{title}', color: '#e6edf3',
                        font: {{ size: 15, family: "'Inter', sans-serif", weight: '600' }},
                        padding: {{ bottom: 16 }}
                    }},
                    tooltip: {{
                        callbacks: {{
                            label: function(ctx) {{
                                if (ctx.dataset.label === 'Seuil maitrise') return null;
                                return ctx.dataset.label + ' : ' + ctx.raw + '%';
                            }}
                        }}
                    }}
                }},
                scales: {{
                    r: {{
                        min: 0, max: 100,
                        ticks: {{ stepSize: 25, color: '#484f58', backdropColor: 'transparent', font: {{ size: 11 }}, callback: function(v) {{ return v + '%'; }} }},
                        grid: {{ color: '#21262d', lineWidth: 1 }},
                        angleLines: {{ color: '#21262d', lineWidth: 1 }},
                        pointLabels: {{ color: '#c9d1d9', font: {{ size: 12, family: "'Inter', sans-serif", weight: '500' }} }}
                    }}
                }}
            }}
        }});
    }})();
    </script>
    """
    components.html(html, height=500)


# bar chart horizontal Chart.js avec ligne cible 70%
def render_chartjs_bar(labels, values, title=""):
    labels_json = json.dumps(labels)
    values_json = json.dumps([round(v * 100, 1) for v in values])
    colors = json.dumps(['#3fb950' if v >= 0.7 else '#d29922' if v >= 0.45 else '#f85149' for v in values])
    safe_id = title.replace(' ', '_').replace("'", "").replace('"', '').replace('—', '')
    bar_h = max(200, len(labels) * 52 + 60)
    html = f"""
    <div style="background:#161b22; border-radius:12px; padding:20px; border:1px solid #21262d;">
        <canvas id="bar_{safe_id}" width="400" height="{bar_h}"></canvas>
        <div style="display:flex;gap:20px;margin-top:8px;font-size:12px;color:#8b949e;">
            <span style="color:#3fb950;">■ Maîtrise (≥70%)</span>
            <span style="color:#d29922;">■ En cours (45-69%)</span>
            <span style="color:#f85149;">■ À développer (&lt;45%)</span>
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3/dist/chartjs-plugin-annotation.min.js"></script>
    <script>
        var ctx = document.getElementById('bar_{safe_id}').getContext('2d');
        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: {labels_json},
                datasets: [{{ data: {values_json}, backgroundColor: {colors}, borderRadius: 6, barThickness: 32 }}]
            }},
            options: {{
                responsive: true, indexAxis: 'y',
                plugins: {{
                    legend: {{ display: false }},
                    title: {{ display: true, text: '{title}', color: '#e6edf3', font: {{ size: 14, family: "'Inter', sans-serif", weight: '600' }} }},
                    annotation: {{
                        annotations: {{
                            cible: {{
                                type: 'line', xMin: 70, xMax: 70,
                                borderColor: '#3fb950', borderWidth: 1.5, borderDash: [6, 4],
                                label: {{ content: 'Seuil 70%', display: true, position: 'start', color: '#3fb950', font: {{ size: 10 }}, backgroundColor: 'transparent' }}
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{ min: 0, max: 100, ticks: {{ color: '#484f58', callback: function(v) {{ return v + '%'; }} }}, grid: {{ color: '#21262d' }} }},
                    y: {{ ticks: {{ color: '#8b949e', font: {{ size: 11 }} }}, grid: {{ display: false }} }}
                }}
            }}
        }});
    </script>
    """
    components.html(html, height=bar_h + 80)


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
<div style="display:flex; align-items:center; gap:12px; margin-bottom:8px; width:100%;">
    {ICONS['target']}
    <h1 style="margin:0; padding:0;">AISCA</h1>
</div>
<p style="color:#8b949e; font-size:0.9rem; margin-top:0; margin-bottom:24px; width:100%;">
    Agent Intelligent de Cartographie des Competences
</p>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# MODE RÉSULTATS
# ─────────────────────────────────────────────────────────────────────
if st.session_state.show_results and st.session_state.results_data:
    rd          = st.session_state.results_data
    top_jobs    = rd["top_jobs"]
    user_text   = rd["user_text"]
    comp_scores = rd["comp_scores"]

    if st.button("← Nouveau questionnaire", use_container_width=False, type="secondary"):
        keys_to_clear = [
            "business", "finance", "design", "communication", "data_analysis",
            "ml", "dev", "engineering", "_juridique_level",
            "rigueur", "leadership", "persuasion", "empathie",
            "projet", "journee", "interet", "defis", "objectif",
            "outils_data", "outils_ml", "outils_dev", "outils_design", "outils_marketing",
            "etudes", "formation", "experience", "secteur",
            "domaines_col1", "domaines_col2", "domaines_col3",
        ]
        for k in keys_to_clear:
            st.session_state.pop(k, None)
        st.session_state.show_results    = False
        st.session_state.results_data    = None
        st.session_state.current_step    = 0
        st.session_state.selected_profile = "-- Aucun (remplir manuellement) --"
        st.session_state.pop("selected_job_idx", None)
        st.rerun()

    st.markdown("---")
    st.markdown(f'<div style="display:flex; align-items:center; gap:10px; margin-bottom:20px;">{ICONS["award"]}<h2 style="margin:0;">Resultats de l\'analyse</h2></div>', unsafe_allow_html=True)

    col1, col2, col3    = st.columns(3)
    overall_score        = top_jobs[0]['score'] if top_jobs else 0
    job_blocs            = top_jobs[0]['blocs']
    nb_fortes            = sum(1 for b in job_blocs if b['score'] >= 0.7)
    nb_total             = len(job_blocs)
    bloc_principal_score = float(job_blocs[0]['score']) if job_blocs else 0.0
    bloc_principal_nom   = job_blocs[0]['nom'] if job_blocs else "—"
    with col1:
        st.metric(f"Adequation — {top_jobs[0]['titre']}", f"{overall_score:.0%}",
                  help="Score de match global entre votre profil et ce metier")
    with col2:
        st.metric("Blocs maitrisés / requis", f"{nb_fortes} / {nb_total}",
                  help=f"Blocs >= 70% parmi les {nb_total} blocs requis par ce metier")
    with col3:
        st.metric("Bloc principal", f"{bloc_principal_score:.0%}",
                  help=f"Score sur {bloc_principal_nom} — le bloc le plus important pour ce metier")

    st.markdown("---")
    st.markdown(f'<div style="display:flex; align-items:center; gap:10px; margin-bottom:20px;">{ICONS["trending"]}<h2 style="margin:0;">Top 3 des metiers recommandes</h2></div>', unsafe_allow_html=True)

    medal_colors = ["#f5a623", "#8b949e", "#cd7f32"]
    medal_labels = ["#1", "#2", "#3"]

    if "selected_job_idx" not in st.session_state:
        st.session_state.selected_job_idx = 0

    cards_html = '<div style="display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:16px;">'
    for i, job in enumerate(top_jobs):
        is_active = (i == st.session_state.selected_job_idx)
        border    = f"2px solid {medal_colors[i]}" if is_active else "1px solid #21262d"
        bg        = "#161b22" if is_active else "#0d1117"
        pct       = int(job['score'] * 100)
        cards_html += f"""
        <div style="background:{bg}; border:{border}; border-radius:12px; padding:16px 20px;">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:10px;">
                <span style="font-size:1.1rem; font-weight:700; color:{medal_colors[i]};">{medal_labels[i]}</span>
                <span style="font-size:0.95rem; font-weight:600; color:#e6edf3; line-height:1.3;">{job['titre']}</span>
            </div>
            <div style="font-size:1.4rem; font-weight:700; color:{medal_colors[i]}; margin-bottom:8px;">{pct}%</div>
            <div style="background:#21262d; border-radius:4px; height:5px;">
                <div style="background:{medal_colors[i]}; border-radius:4px; height:5px; width:{pct}%;"></div>
            </div>
        </div>"""
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    sel_cols = st.columns(3)
    for i, (col, job) in enumerate(zip(sel_cols, top_jobs)):
        with col:
            label = f"✓ #{i+1} sélectionné" if i == st.session_state.selected_job_idx else f"Voir #{i+1}"
            if st.button(label, key=f"job_select_{i}", use_container_width=True,
                         type="primary" if i == st.session_state.selected_job_idx else "secondary"):
                st.session_state.selected_job_idx = i
                st.rerun()

    job          = top_jobs[st.session_state.selected_job_idx]
    i_job        = st.session_state.selected_job_idx
    global_score = float(job['score'])

    st.markdown(f"""
    <div style="background:#161b22; border:1px solid #21262d; border-left:4px solid {medal_colors[i_job]};
                border-radius:12px; padding:20px 24px; margin:20px 0 8px 0;">
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:4px;">
            <span style="font-size:1.4rem; font-weight:700; color:{medal_colors[i_job]};">#{i_job+1}</span>
            <span style="font-size:1.25rem; font-weight:700; color:#e6edf3;">{job['titre']}</span>
            <span style="margin-left:auto; font-size:1.6rem; font-weight:800; color:{medal_colors[i_job]};">{global_score:.0%}</span>
        </div>
        {'<p style="color:#8b949e; font-size:0.88rem; margin:0;">' + job['description'] + '</p>' if job.get('description') else ''}
    </div>
    """, unsafe_allow_html=True)

    col_chart, col_detail = st.columns([3, 2])

    with col_chart:
        if job.get('blocs') and len(job['blocs']) > 0:
            labels = [b['nom'] for b in job['blocs']]
            values = [float(b['score']) for b in job['blocs']]
            if len(labels) >= 3:
                render_chartjs_radar(labels, values, f"Cartographie — {job['titre']}")
            else:
                render_chartjs_bar(labels, values, f"Adequation — {job['titre']}")

    with col_detail:
        st.markdown("#### Compétences par bloc")
        # Pour chaque bloc requis : points forts (≥60%) et compétences à développer (<60%)
        # issues directement des scores individuels comp_scores
        for bloc in job['blocs']:
            b_score = float(bloc['score'])
            b_nom   = bloc['nom']
            b_label = "Maîtrise" if b_score >= 0.70 else "En cours" if b_score >= 0.45 else "À développer"

            bloc_comps = sorted([
                (comp_idx[cid]['texte'], score)
                for cid, score in comp_scores.items()
                if cid in comp_idx and comp_idx[cid]['bloc_nom'] == b_nom
            ], key=lambda x: -x[1])

            with st.expander(f"{b_nom}  —  {b_score:.0%}  ·  {b_label}", expanded=(b_score < 0.70)):
                acquises = [c for c in bloc_comps if c[1] >= 0.60][:3]
                a_dev    = [c for c in bloc_comps if c[1] < 0.60][:4]

                if acquises:
                    st.markdown('<p style="color:#3fb950; font-size:0.8rem; font-weight:600; margin:0 0 6px;">✓ Points forts</p>', unsafe_allow_html=True)
                    for texte, score in acquises:
                        st.markdown(f"""
                        <div style="display:flex; justify-content:space-between; align-items:center;
                                    padding:5px 0; border-bottom:1px solid #21262d; font-size:0.82rem;">
                            <span style="color:#e6edf3; flex:1; padding-right:10px;">{texte}</span>
                            <span style="color:#3fb950; font-weight:600; white-space:nowrap;">{score:.0%}</span>
                        </div>""", unsafe_allow_html=True)

                if a_dev:
                    st.markdown('<p style="color:#f85149; font-size:0.8rem; font-weight:600; margin:10px 0 6px;">↑ À développer</p>', unsafe_allow_html=True)
                    for texte, score in a_dev:
                        st.markdown(f"""
                        <div style="display:flex; justify-content:space-between; align-items:center;
                                    padding:5px 0; border-bottom:1px solid #21262d; font-size:0.82rem;">
                            <span style="color:#8b949e; flex:1; padding-right:10px;">{texte}</span>
                            <span style="color:#f85149; font-weight:600; white-space:nowrap;">{score:.0%}</span>
                        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f'<div style="display:flex; align-items:center; gap:10px; margin-bottom:16px;">{ICONS["book"]}<h2 style="margin:0;">Recommandations</h2></div>', unsafe_allow_html=True)

    job_sel      = top_jobs[0]
    weak_blocs   = {b['nom']: b['score'] for b in job_sel['blocs'] if b['score'] < 0.6}
    strong_blocs = {b['nom']: b['score'] for b in job_sel['blocs'] if b['score'] >= 0.6}

    col_plan, col_bio = st.columns([3, 2])

    with col_plan:
        st.markdown("#### Plan de progression personnalisé")

        if weak_blocs:
            with st.spinner("Generation du plan..."):
                # compétences individuelles les plus faibles pour un plan concret
                comps_a_dev = []
                for b in job_sel['blocs']:
                    if b['score'] < 0.6:
                        blk_comps = sorted([
                            (comp_idx[cid]['texte'], s)
                            for cid, s in comp_scores.items()
                            if cid in comp_idx and comp_idx[cid]['bloc_nom'] == b['nom']
                        ], key=lambda x: x[1])[:3]
                        comps_a_dev.extend([t for t, _ in blk_comps])

                # on passe tout le profil brut + compétences spécifiques à Gemini
                user_profile_data = {
                    "likert_levels": rd.get("likert_levels", {}),
                    "outils":        rd.get("outils", {}),
                    "soft_skills":   rd.get("soft_skills", {}),
                    "textes_libres": rd.get("textes_libres", {}),
                    "formation":     rd.get("formation", {}),
                    "comps_a_dev":   comps_a_dev[:8],
                }
                learning_path = generate_learning_path(
                    weak_blocs, strong_blocs, job_sel['titre'], user_profile_data
                )
            st.markdown(learning_path)
        else:
            st.success("Excellent ! Vous maitrisez deja tous les blocs requis pour ce metier.")

    with col_bio:
        st.markdown("#### Bio professionnelle")
        if not strong_blocs:
            best = max(job_sel['blocs'], key=lambda b: b['score'])
            strong_blocs = {best['nom']: best['score']}
        with st.spinner("Redaction..."):
            bio = generate_professional_bio(
                strong_blocs, job_sel['titre'],
                {"projet_tech": rd.get("projet_tech", ""), "objectif": rd.get("objectif", "")}
            )
        st.info(bio)
        st.download_button("Télécharger ma bio", bio, file_name="bio_aisca.txt",
                           mime="text/plain", use_container_width=True)

    st.markdown("---")
    st.markdown(f"**Tableau récapitulatif** — blocs requis pour : *{job_sel['titre']}*")
    df_scores = pd.DataFrame([
        {
            "Rang":     f"#{idx+1}",
            "Bloc":     b['nom'],
            "Score":    f"{b['score']:.0%}",
            "Niveau":   ("✓ Maîtrise" if b['score'] >= 0.70 else "~ En cours" if b['score'] >= 0.45 else "↑ À développer"),
            "Priorité": ("Bloc principal" if idx == 0 else "Bloc secondaire" if idx == 1 else "Bloc complémentaire")
        }
        for idx, b in enumerate(job_sel['blocs'])
    ])
    st.dataframe(df_scores, use_container_width=True, hide_index=True)

    with st.expander("Détails techniques", expanded=False):
        st.write(f"**Mots :** {len(user_text.split())} | **Caractères :** {len(user_text)}")
        st.text_area("Texte envoyé au modèle :", user_text, height=150)
        if comp_scores:
            top_comps = sorted(comp_scores.items(), key=lambda x: -x[1])[:10]
            st.write("**Top 10 compétences matchées :**")
            for cid, score in top_comps:
                if cid in comp_idx:
                    st.write(f"- {comp_idx[cid]['texte']} : {score:.2%}")

# ─────────────────────────────────────────────────────────────────────
# MODE QUESTIONNAIRE (wizard 6 étapes, widgets libres sans st.form)
# ─────────────────────────────────────────────────────────────────────
else:
    # Sans st.form, chaque widget écrit directement dans session_state.
    # Les profils de test se préremplissent immédiatement : pas de snapshot figé.

    if "selected_profile" not in st.session_state:
        st.session_state.selected_profile = "-- Aucun (remplir manuellement) --"

    selected_profile = st.selectbox(
        "Profil de test (optionnel)",
        list(TEST_PROFILES.keys()),
        index=list(TEST_PROFILES.keys()).index(st.session_state.selected_profile),
        help="Selectionner un profil pre-rempli pour la demo"
    )

    if selected_profile != st.session_state.selected_profile:
        st.session_state.selected_profile = selected_profile
        if selected_profile != "-- Aucun (remplir manuellement) --":
            profile = TEST_PROFILES[selected_profile]
            if profile:
                st.session_state.pop("_juridique_level", None)
                for k, v in profile.items():
                    st.session_state[k] = v
                st.session_state.current_step = 0
        else:
            st.session_state.pop("_juridique_level", None)
        st.rerun()

    current = st.session_state.current_step
    render_progress_bar(current, len(STEP_NAMES), STEP_NAMES)

    with st.container(border=True):

        if current == 0:
            render_step_header(1, "Domaines d'interet", "Selectionnez les domaines qui vous attirent", "compass")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.multiselect("Business & juridique",
                    ["Juridique", "Business et Stratégie", "Marketing et Vente", "Finance et Comptabilité"],
                    default=st.session_state.get("domaines_col1", []), key="domaines_col1")
            with col2:
                st.multiselect("Creatif & communication",
                    ["Communication et Médias", "Création et Design", "Digital et Réseaux Sociaux"],
                    default=st.session_state.get("domaines_col2", []), key="domaines_col2")
            with col3:
                st.multiselect("Technique & data",
                    ["Data Analysis", "Machine Learning et IA", "Développement et Infrastructure", "Ingénierie et Technique"],
                    default=st.session_state.get("domaines_col3", []), key="domaines_col3")

        elif current == 1:
            render_step_header(2, "Auto-evaluation technique", "Evaluez votre niveau dans chaque domaine", "sliders")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Business & Strategie**")
                st.select_slider("Strategie d'entreprise",
                    options=["Débutant", "Intermédiaire", "Avancé", "Expert"],
                    value=st.session_state.get("business", "Débutant"), key="business")
                st.select_slider("Finance / Comptabilite",
                    options=["Débutant", "Intermédiaire", "Avancé", "Expert"],
                    value=st.session_state.get("finance", "Débutant"), key="finance")
                st.markdown("**Data & Analyse**")
                st.select_slider("Analyse de donnees",
                    options=["Débutant", "Intermédiaire", "Avancé", "Expert"],
                    value=st.session_state.get("data_analysis", "Débutant"), key="data_analysis")
                st.select_slider("Machine Learning / IA",
                    options=["Débutant", "Intermédiaire", "Avancé", "Expert"],
                    value=st.session_state.get("ml", "Débutant"), key="ml")
            with col2:
                st.markdown("**Creativite & Communication**")
                st.select_slider("Design / Creativite",
                    options=["Débutant", "Intermédiaire", "Avancé", "Expert"],
                    value=st.session_state.get("design", "Débutant"), key="design")
                st.select_slider("Communication / Media",
                    options=["Débutant", "Intermédiaire", "Avancé", "Expert"],
                    value=st.session_state.get("communication", "Débutant"), key="communication")
                st.markdown("**Developpement & Tech**")
                st.select_slider("Developpement logiciel",
                    options=["Débutant", "Intermédiaire", "Avancé", "Expert"],
                    value=st.session_state.get("dev", "Débutant"), key="dev")
                st.select_slider("Ingenierie technique",
                    options=["Débutant", "Intermédiaire", "Avancé", "Expert"],
                    value=st.session_state.get("engineering", "Débutant"), key="engineering")

        elif current == 2:
            render_step_header(3, "Soft skills", "Vos traits de personnalite", "users")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Style de travail**")
                _rigueur_opts = ["Rigoureux et attentif aux détails", "Créatif et improvisateur", "Équilibré entre les deux"]
                _rigueur_val  = st.session_state.get("rigueur", _rigueur_opts[0])
                if _rigueur_val not in _rigueur_opts: _rigueur_val = _rigueur_opts[0]
                st.radio("Vous etes plutot :", _rigueur_opts, index=_rigueur_opts.index(_rigueur_val), key="rigueur")

                _lead_opts = ["Je prends naturellement le lead", "Je préfère contribuer en tant que membre", "Cela dépend du contexte"]
                _lead_val  = st.session_state.get("leadership", _lead_opts[0])
                if _lead_val not in _lead_opts: _lead_val = _lead_opts[0]
                st.radio("Face a un projet d'equipe :", _lead_opts, index=_lead_opts.index(_lead_val), key="leadership")
            with col2:
                st.markdown("**Competences relationnelles**")
                _pers_opts = ["Oui, j'adore argumenter et persuader", "Non, je préfère éviter les confrontations", "Parfois, selon le sujet"]
                _pers_val  = st.session_state.get("persuasion", _pers_opts[0])
                if _pers_val not in _pers_opts: _pers_val = _pers_opts[0]
                st.radio("Debattre et convaincre ?", _pers_opts, index=_pers_opts.index(_pers_val), key="persuasion")

                _empa_opts = ["Je suis très empathique et à l'écoute", "Je privilégie l'efficacité sur l'émotion", "Je trouve un équilibre"]
                _empa_val  = st.session_state.get("empathie", _empa_opts[0])
                if _empa_val not in _empa_opts: _empa_val = _empa_opts[0]
                st.radio("Relations professionnelles :", _empa_opts, index=_empa_opts.index(_empa_val), key="empathie")

        elif current == 3:
            render_step_header(4, "Experiences et aspirations", "Detaillez pour une analyse plus precise", "edit")
            st.text_area("Un projet dont vous etes fier",
                placeholder="Ex : J'ai developpe un dashboard Power BI...",
                value=st.session_state.get("projet", ""), height=100, key="projet")
            col1, col2 = st.columns(2)
            with col1:
                st.text_area("Journee de travail ideale",
                    placeholder="Ex : Alterner entre analyse de donnees...",
                    value=st.session_state.get("journee", ""), height=90, key="journee")
                st.text_area("Domaines qui vous passionnent",
                    placeholder="Ex : L'ethique de l'IA...",
                    value=st.session_state.get("interet", ""), height=90, key="interet")
            with col2:
                st.text_area("Defis qui vous stimulent",
                    placeholder="Ex : Resoudre des problemes complexes...",
                    value=st.session_state.get("defis", ""), height=90, key="defis")
                st.text_area("Objectif de carriere a 3-5 ans",
                    placeholder="Ex : Devenir Lead Data Scientist...",
                    value=st.session_state.get("objectif", ""), height=90, key="objectif")

        elif current == 4:
            render_step_header(5, "Outils et technologies", "Selectionnez ce que vous maitrisez", "wrench")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Data & Analytics**")
                st.multiselect("Outils data",
                    ["Python", "R", "SQL", "Excel avancé", "Pandas", "NumPy", "Matplotlib", "Seaborn", "Plotly", "Tableau", "Power BI"],
                    default=st.session_state.get("outils_data", []), key="outils_data")
            with col2:
                st.markdown("**IA & Machine Learning**")
                st.multiselect("Frameworks IA",
                    ["Scikit-learn", "TensorFlow", "PyTorch", "Keras", "HuggingFace", "LangChain", "OpenAI API", "BERT", "GPT"],
                    default=st.session_state.get("outils_ml", []), key="outils_ml")
            with col3:
                st.markdown("**Dev & Cloud**")
                st.multiselect("Stack technique",
                    ["Git", "Docker", "Kubernetes", "AWS", "Azure", "GCP", "Flask", "Django", "FastAPI", "React", "Node.js", "MongoDB"],
                    default=st.session_state.get("outils_dev", []), key="outils_dev")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Design & Creativite**")
                st.multiselect("Suite creative",
                    ["Photoshop", "Illustrator", "InDesign", "Figma", "Canva", "After Effects", "Premiere Pro", "Sketch"],
                    default=st.session_state.get("outils_design", []), key="outils_design")
            with col2:
                st.markdown("**Marketing & Communication**")
                st.multiselect("Outils marketing",
                    ["Google Analytics", "Meta Business Suite", "Google Ads", "SEMrush", "Mailchimp", "HubSpot", "Hootsuite", "Buffer"],
                    default=st.session_state.get("outils_marketing", []), key="outils_marketing")

        elif current == 5:
            render_step_header(6, "Parcours et experience", "Formation et experience professionnelle", "briefcase")
            col1, col2 = st.columns(2)
            _etudes_opts = ["Bac", "Bac+2 (BTS/DUT)", "Bac+3 (Licence)", "Bac+5 (Master)", "Bac+8 (Doctorat)", "Autre"]
            _exp_opts    = ["Étudiant / 0 an", "0-2 ans", "3-5 ans", "6-10 ans", "+10 ans"]
            _etudes_val  = st.session_state.get("etudes", _etudes_opts[0])
            _exp_val     = st.session_state.get("experience", _exp_opts[0])
            if _etudes_val not in _etudes_opts: _etudes_val = _etudes_opts[0]
            if _exp_val    not in _exp_opts:    _exp_val    = _exp_opts[0]
            with col1:
                st.selectbox("Niveau d'etudes", _etudes_opts,
                    index=_etudes_opts.index(_etudes_val), key="etudes")
                st.text_input("Domaine de formation",
                    placeholder="Ex : Data Science, Marketing...",
                    value=st.session_state.get("formation", ""), key="formation")
            with col2:
                st.selectbox("Experience professionnelle", _exp_opts,
                    index=_exp_opts.index(_exp_val), key="experience")
                st.text_input("Secteur d'activite",
                    placeholder="Ex : Banque, E-commerce...",
                    value=st.session_state.get("secteur", ""), key="secteur")

    col_prev, col_spacer, col_next = st.columns([1, 2, 1])
    with col_prev:
        if current > 0:
            if st.button("Précédent", use_container_width=True):
                st.session_state.current_step -= 1
                st.rerun()
    with col_next:
        if current < 5:
            if st.button("Suivant", use_container_width=True, type="primary"):
                st.session_state.current_step += 1
                st.rerun()
        else:
            if st.button("Analyser mon profil", use_container_width=True, type="primary"):

                likert_levels = {
                    "business":      st.session_state.get("business",      "Débutant"),
                    "finance":       st.session_state.get("finance",       "Débutant"),
                    "design":        st.session_state.get("design",        "Débutant"),
                    "communication": st.session_state.get("communication", "Débutant"),
                    "data_analysis": st.session_state.get("data_analysis", "Débutant"),
                    "ml":            st.session_state.get("ml",            "Débutant"),
                    "dev":           st.session_state.get("dev",           "Débutant"),
                    "engineering":   st.session_state.get("engineering",   "Débutant"),
                }
                juridique_level = st.session_state.get("_juridique_level", None)
                if juridique_level:
                    likert_levels["juridique"] = juridique_level

                _projet   = st.session_state.get("projet",   "")
                _interet  = st.session_state.get("interet",  "")
                _objectif = st.session_state.get("objectif", "")
                _journee  = st.session_state.get("journee",  "")
                _defis    = st.session_state.get("defis",    "")

                outils_all = (
                    st.session_state.get("outils_data",      []) +
                    st.session_state.get("outils_ml",        []) +
                    st.session_state.get("outils_dev",       []) +
                    st.session_state.get("outils_design",    []) +
                    st.session_state.get("outils_marketing", [])
                )
                outils_text        = f"Je maîtrise : {', '.join(outils_all)}." if outils_all else ""
                background_summary = (
                    f"Formation : {st.session_state.get('etudes', '')} "
                    f"en {st.session_state.get('formation', '')}. "
                    f"Experience : {st.session_state.get('experience', '')} "
                    f"dans le secteur {st.session_state.get('secteur', '')}."
                )
                soft_skills_summary = (
                    f"Style de travail : {st.session_state.get('rigueur', '')}. "
                    f"Leadership : {st.session_state.get('leadership', '')}. "
                    f"Persuasion : {st.session_state.get('persuasion', '')}. "
                    f"Empathie : {st.session_state.get('empathie', '')}."
                )

                extra_text = " ".join(filter(None, [
                    soft_skills_summary,
                    _projet, _journee, _interet, _defis, _objectif,
                    outils_text, background_summary
                ])).strip()

                if len(extra_text.split()) < 10 and all(v == "Débutant" for v in likert_levels.values()):
                    st.warning("Reponses trop courtes. Evaluez au moins un domaine et detaillez vos experiences.")
                else:
                    with st.spinner("Pretraitement semantique..."):
                        enriched_projet  = enrich_short_text(_projet)  if _projet  and len(_projet.split())  < 15 else _projet
                        enriched_interet = enrich_short_text(_interet) if _interet and len(_interet.split()) < 10 else _interet
                        enriched_extra   = extra_text.replace(_projet, enriched_projet).replace(_interet, enriched_interet)

                    with st.spinner("Analyse NLP en cours..."):
                        user_emb = build_profile_embedding(
                            levels=likert_levels,
                            extra_text=enriched_extra
                        )
                        comp_scores = analyze_profile(
                            user_emb, bi_model, comp_ids, embeddings, None, comp_idx, top_k=None
                        )
                        top_jobs, bloc_scores = recommend_jobs(comp_scores, data, top_n=3)

                    st.session_state.results_data = {
                        "top_jobs":    top_jobs,
                        "bloc_scores": bloc_scores,
                        "user_text":   enriched_extra,
                        "final_text":  enriched_extra,
                        "comp_scores": comp_scores,
                        "projet_tech": _projet,
                        "objectif":    _objectif,
                        # profil brut structuré — utilisé par generate_learning_path
                        # pour un plan de progression concret et personnalisé
                        "likert_levels": dict(likert_levels),
                        "outils": {
                            "Data & Analytics":          st.session_state.get("outils_data",      []),
                            "IA & Machine Learning":     st.session_state.get("outils_ml",        []),
                            "Dev & Cloud":               st.session_state.get("outils_dev",       []),
                            "Design & Creativite":       st.session_state.get("outils_design",    []),
                            "Marketing & Communication": st.session_state.get("outils_marketing", []),
                        },
                        "soft_skills": {
                            "Rigueur / Creativite": st.session_state.get("rigueur",    ""),
                            "Leadership":           st.session_state.get("leadership", ""),
                            "Persuasion":           st.session_state.get("persuasion", ""),
                            "Empathie":             st.session_state.get("empathie",   ""),
                        },
                        "textes_libres": {
                            "Projet dont il est fier":     _projet,
                            "Journee ideale":              _journee,
                            "Domaines qui le passionnent": _interet,
                            "Defis qui le stimulent":      _defis,
                            "Objectif de carriere":        _objectif,
                        },
                        "formation": {
                            "Niveau d'etudes":            st.session_state.get("etudes",     ""),
                            "Domaine de formation":       st.session_state.get("formation",  ""),
                            "Experience professionnelle": st.session_state.get("experience", ""),
                            "Secteur d'activite":         st.session_state.get("secteur",    ""),
                        },
                    }
                    st.session_state.show_results = True
                    st.rerun()

# footer
st.markdown("---")
st.markdown('<p style="color:#484f58; font-size:0.75rem; text-align:center;">AISCA - 2026 | EFREI - Master Data Engineering & AI | Projet IA Generative</p>', unsafe_allow_html=True)