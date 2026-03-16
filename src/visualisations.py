# ancienne version de l'interface streamlit, on l'a gardée en backup
# la version actuelle c'est test-visualisations.py
# celle-ci est quasiment identique mais sans les dernières corrections

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

from scoring import get_models, load_and_index_data, analyze_profile, recommend_jobs, likert_to_semantic_text
from genai_augmentation import enrich_short_text, generate_learning_path, generate_professional_bio

st.set_page_config(
    page_title="OSCC",
    page_icon="favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_PATH = "data/referentiel.json"


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


with st.sidebar:
    st.image("src/visuel/img/projet_IA_Generative.png", width="stretch")
    st.header("Agent RAG Intelligent")
    st.markdown("""
    **Pipeline d'analyse :**
    1. Retrieval : Matching semantique SBERT multilingue
    2. Augmentation : Enrichissement GenAI
    3. Generation : Recommandations personnalisees

    11 blocs de competences
    40 metiers analyses
    200+ competences indexees
    """)

    st.divider()
    st.caption("Stack technique :")
    st.code("NLP: paraphrase-multilingual-mpnet-base-v2\nGenAI: Gemini 2.5 Flash", language="text")

with st.sidebar:
    st.divider()
    st.header("Profils de Test")
    st.caption("Cliquez pour remplir automatiquement le formulaire")

    # profil data scientist
    if st.button("Data Scientist", width="stretch"):
        st.session_state.update({
            "business":      "Débutant",
            "finance":       "Débutant",
            "design":        "Débutant",
            "communication": "Débutant",
            "data_analysis": "Expert",
            "ml":            "Expert",
            "dev":           "Avancé",
            "engineering":   "Débutant",
            "rigueur":    "Rigoureux et attentif aux détails",
            "leadership": "Je préfère contribuer en tant que membre",
            "persuasion": "Parfois, selon le sujet",
            "empathie":   "Je trouve un équilibre",
            "projet":  "J'ai développé un système de prédiction de churn client avec Python, scikit-learn et XGBoost. Modèles de classification supervisée à 92% de précision, déployés via FastAPI et Docker avec pipelines MLOps.",
            "journee": "Analyser des datasets complexes, entraîner des modèles de machine learning, évaluer leurs performances et déployer des pipelines de données en production.",
            "interet": "Machine Learning, Deep Learning, NLP, transformers, modèles prédictifs, TensorFlow, PyTorch, HuggingFace.",
            "defis":   "Optimiser des algorithmes de machine learning, améliorer la précision des modèles, trouver des patterns dans des datasets massifs.",
            "objectif":"Devenir Lead Data Scientist ou ML Engineer spécialisé en NLP et déploiement de modèles IA en production.",
            "outils_data":     ["Python", "Pandas", "SQL", "Matplotlib", "NumPy", "Seaborn"],
            "outils_ml":       ["Scikit-learn", "TensorFlow", "PyTorch", "HuggingFace", "BERT"],
            "outils_dev":      ["Git", "Docker", "FastAPI"],
            "outils_design":   [],
            "outils_marketing":[],
            "formation": "Data Science et Machine Learning",
            "etudes":    "Bac+5 (Master)",
            "experience":"3-5 ans",
            "secteur":   "Intelligence Artificielle et Data"
        })
        st.rerun()

    # profil marketing / com
    if st.button("Marketing / Communication", width="stretch"):
        st.session_state.update({
            "business":      "Intermédiaire",
            "finance":       "Débutant",
            "design":        "Intermédiaire",
            "communication": "Expert",
            "data_analysis": "Débutant",
            "ml":            "Débutant",
            "dev":           "Débutant",
            "engineering":   "Débutant",
            "rigueur":    "Créatif et improvisateur",
            "leadership": "Je prends naturellement le lead",
            "persuasion": "Oui, j'adore argumenter et persuader",
            "empathie":   "Je suis très empathique et à l'écoute",
            "projet":  "J'ai piloté la stratégie de communication d'une startup : refonte de l'identité de marque, gestion des réseaux sociaux Instagram et LinkedIn, campagnes publicitaires Meta Ads avec +40% d'engagement. Organisation d'événements presse et partenariats influenceurs.",
            "journee": "Créer du contenu engageant pour les réseaux sociaux, rédiger des communiqués de presse, analyser les performances des campagnes et coordonner les équipes créatives.",
            "interet": "Stratégie de marque, storytelling, réseaux sociaux Instagram LinkedIn TikTok, relations presse, influence marketing, content marketing, community management, campagnes Meta Ads Google Ads, SEO SEA.",
            "defis":   "Construire une image de marque forte, toucher des audiences ciblées et mesurer l'impact des campagnes de communication.",
            "objectif":"Devenir Directrice de la Communication ou Social Media Manager dans une agence créative internationale.",
            "outils_data":     ["Excel avancé"],
            "outils_ml":       [],
            "outils_dev":      [],
            "outils_design":   ["Canva", "InDesign", "Photoshop"],
            "outils_marketing":["Meta Business Suite", "Google Analytics", "Hootsuite", "Mailchimp", "HubSpot"],
            "formation": "Communication et Marketing Digital",
            "etudes":    "Bac+5 (Master)",
            "experience":"3-5 ans",
            "secteur":   "Agence de communication et marketing"
        })
        st.rerun()

    # profil designer
    if st.button("Designer / Creatif", width="stretch"):
        st.session_state.update({
            "business":      "Débutant",
            "finance":       "Débutant",
            "design":        "Expert",
            "communication": "Avancé",
            "data_analysis": "Débutant",
            "ml":            "Débutant",
            "dev":           "Intermédiaire",
            "engineering":   "Débutant",
            "rigueur":    "Créatif et improvisateur",
            "leadership": "Cela dépend du contexte",
            "persuasion": "Parfois, selon le sujet",
            "empathie":   "Je suis très empathique et à l'écoute",
            "projet":  "J'ai conçu l'identité visuelle complète d'une marque de cosmétiques : logo, charte graphique, packaging, site web et supports print. Direction artistique de la campagne de lancement avec +200k impressions.",
            "journee": "Concevoir des visuels créatifs, travailler sur des mises en page print et digital, collaborer avec les équipes marketing pour traduire une vision en identité visuelle forte.",
            "interet": "Design graphique, direction artistique, typographie, identité visuelle, motion design, illustration, expérience utilisateur UI/UX.",
            "defis":   "Créer des identités visuelles mémorables, trouver l'équilibre entre esthétique et communication, innover dans les tendances design.",
            "objectif":"Devenir Directeur Artistique ou UX/UI Designer Lead dans une agence de design internationale.",
            "outils_data":     [],
            "outils_ml":       [],
            "outils_dev":      ["Git"],
            "outils_design":   ["Photoshop", "Illustrator", "InDesign", "Figma", "After Effects"],
            "outils_marketing":["Google Analytics"],
            "formation": "Design Graphique et Direction Artistique",
            "etudes":    "Bac+5 (Master)",
            "experience":"3-5 ans",
            "secteur":   "Agence créative et design"
        })
        st.rerun()

    # profil juriste, on injecte le niveau juridique via _juridique_level
    if st.button("Juriste / Consultant", width="stretch"):
        st.session_state.update({
            "business":      "Intermédiaire",
            "finance":       "Intermédiaire",
            "design":        "Débutant",
            "communication": "Avancé",
            "data_analysis": "Débutant",
            "ml":            "Débutant",
            "dev":           "Débutant",
            "engineering":   "Débutant",
            "_juridique_level": "Expert",
            "rigueur":    "Rigoureux et attentif aux détails",
            "leadership": "Je prends naturellement le lead",
            "persuasion": "Oui, j'adore argumenter et persuader",
            "empathie":   "Je trouve un équilibre",
            "projet":  "J'ai accompagné une PME dans sa mise en conformité RGPD : audit des traitements de données personnelles, rédaction des contrats DPA et des actes juridiques, formation des équipes et dépôt des registres CNIL. Rédaction de contrats commerciaux complexes en droit des affaires et droit du travail.",
            "journee": "Analyser des contrats et textes juridiques complexes, rédiger des actes juridiques et documents légaux, conseiller sur les risques réglementaires, effectuer une veille juridique continue sur le droit des affaires et la conformité RGPD.",
            "interet": "Droit des affaires, droit des contrats, conformité RGPD, propriété intellectuelle, droit du travail, contentieux, réglementation, actes juridiques.",
            "defis":   "Sécuriser juridiquement les opérations d'entreprise, interpréter des textes juridiques complexes, anticiper les risques réglementaires et défendre les intérêts des clients.",
            "objectif":"Devenir Juriste d'entreprise Senior ou Avocat d'affaires spécialisé en droit du numérique et conformité RGPD.",
            "outils_data":     ["Excel avancé"],
            "outils_ml":       [],
            "outils_dev":      [],
            "outils_design":   [],
            "outils_marketing":[],
            "formation": "Droit des affaires et Conformité réglementaire",
            "etudes":    "Bac+5 (Master)",
            "experience":"3-5 ans",
            "secteur":   "Cabinet d'avocats et conseil juridique"
        })
        st.rerun()


st.title("AISCA - Agent Intelligent de Cartographie des Competences")
st.markdown("*Analyse semantique avancee pour l'orientation professionnelle*")

st.divider()

st.header("Questionnaire d'evaluation des competences")
st.markdown("Repondez aux questions suivantes pour obtenir votre profil personnalise.")

with st.form(key="questionnaire_complet"):

    st.subheader("1. Vos domaines d'interet professionnels")
    st.caption("Selectionnez tous les domaines qui vous attirent")

    col1, col2, col3 = st.columns(3)

    with col1:
        domaines_col1 = st.multiselect(
            "Domaines business et juridique :",
            ["Juridique", "Business et Stratégie", "Marketing et Vente", "Finance et Comptabilité"],
            help="Compétences en droit, stratégie, commerce et finance"
        )

    with col2:
        domaines_col2 = st.multiselect(
            "Domaines creatifs et communication :",
            ["Communication et Médias", "Création et Design", "Digital et Réseaux Sociaux"],
            help="Compétences en communication, design et social media"
        )

    with col3:
        domaines_col3 = st.multiselect(
            "Domaines techniques et data :",
            ["Data Analysis", "Machine Learning et IA", "Développement et Infrastructure", "Ingénierie et Technique"],
            help="Compétences en data, IA, développement et ingénierie"
        )

    domaines = domaines_col1 + domaines_col2 + domaines_col3

    st.divider()

    st.subheader("2. Auto-evaluation de vos competences techniques")
    st.caption("Evaluez honnetement votre niveau actuel dans chaque domaine")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**Business et Strategie**")
        business_level = st.select_slider(
            "Stratégie d'entreprise",
            options=["Débutant", "Intermédiaire", "Avancé", "Expert"],
            value="Débutant",
            key="business"
        )
        finance_level = st.select_slider(
            "Finance et Comptabilite",
            options=["Débutant", "Intermédiaire", "Avancé", "Expert"],
            value="Débutant",
            key="finance"
        )

    with col2:
        st.markdown("**Creativite et Communication**")
        design_level = st.select_slider(
            "Design et Creativite",
            options=["Débutant", "Intermédiaire", "Avancé", "Expert"],
            value="Débutant",
            key="design"
        )
        communication_level = st.select_slider(
            "Communication et Media",
            options=["Débutant", "Intermédiaire", "Avancé", "Expert"],
            value="Débutant",
            key="communication"
        )

    with col3:
        st.markdown("**Data et Analyse**")
        data_analysis_level = st.select_slider(
            "Analyse de donnees",
            options=["Débutant", "Intermédiaire", "Avancé", "Expert"],
            value="Débutant",
            key="data_analysis"
        )
        ml_level = st.select_slider(
            "Machine Learning et IA",
            options=["Débutant", "Intermédiaire", "Avancé", "Expert"],
            value="Débutant",
            key="ml"
        )

    with col4:
        st.markdown("**Developpement et Tech**")
        dev_level = st.select_slider(
            "Developpement logiciel",
            options=["Débutant", "Intermédiaire", "Avancé", "Expert"],
            value="Débutant",
            key="dev"
        )
        engineering_level = st.select_slider(
            "Ingenierie technique",
            options=["Débutant", "Intermédiaire", "Avancé", "Expert"],
            value="Débutant",
            key="engineering"
        )

    st.divider()

    st.subheader("3. Traits de personnalite et soft skills")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Style de travail**")
        rigoeur = st.radio(
            "Vous etes plutot :",
            ["Rigoureux et attentif aux détails", "Créatif et improvisateur", "Équilibré entre les deux"],
            key="rigueur"
        )
        leadership = st.radio(
            "Face a un projet d'equipe :",
            ["Je prends naturellement le lead", "Je préfère contribuer en tant que membre", "Cela dépend du contexte"],
            key="leadership"
        )

    with col2:
        st.markdown("**Competences relationnelles**")
        persuasion = st.radio(
            "Aimez-vous debattre et convaincre ?",
            ["Oui, j'adore argumenter et persuader", "Non, je préfère éviter les confrontations", "Parfois, selon le sujet"],
            key="persuasion"
        )
        empathie = st.radio(
            "Dans vos relations professionnelles :",
            ["Je suis très empathique et à l'écoute", "Je privilégie l'efficacité sur l'émotion", "Je trouve un équilibre"],
            key="empathie"
        )

    st.divider()

    st.subheader("4. Experiences et aspirations")
    st.caption("Detaillez vos reponses pour une analyse plus precise")

    projet_tech = st.text_area(
        "Decrivez un projet technique ou professionnel dont vous etes fier :",
        placeholder="Ex : J'ai développé un dashboard Power BI pour analyser les ventes en temps réel...",
        height=120,
        key="projet"
    )

    col1, col2 = st.columns(2)

    with col1:
        journee_ideale = st.text_area(
            "Decrivez votre journee de travail ideale :",
            placeholder="Ex : Alterner entre analyse de données, réunions stratégiques...",
            height=100,
            key="journee"
        )
        interet_specifique = st.text_area(
            "Domaines specifiques qui vous passionnent :",
            placeholder="Ex : L'éthique de l'IA, l'optimisation énergétique...",
            height=100,
            key="interet"
        )

    with col2:
        defis_aimes = st.text_area(
            "Types de defis qui vous stimulent :",
            placeholder="Ex : Résoudre des problèmes complexes avec la data...",
            height=100,
            key="defis"
        )
        objectif_carriere = st.text_area(
            "Objectif de carriere a 3-5 ans :",
            placeholder="Ex : Devenir Lead Data Scientist dans une fintech...",
            height=100,
            key="objectif"
        )

    st.divider()

    st.subheader("5. Outils et technologies maitrises")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Data et Analytics**")
        outils_data = st.multiselect(
            "Outils maitrises :",
            ["Python", "R", "SQL", "Excel avancé", "Pandas", "NumPy", "Matplotlib", "Seaborn", "Plotly", "Tableau", "Power BI"],
            key="outils_data"
        )

    with col2:
        st.markdown("**IA et Machine Learning**")
        outils_ml = st.multiselect(
            "Frameworks IA :",
            ["Scikit-learn", "TensorFlow", "PyTorch", "Keras", "HuggingFace", "LangChain", "OpenAI API", "BERT", "GPT"],
            key="outils_ml"
        )

    with col3:
        st.markdown("**Developpement et Cloud**")
        outils_dev = st.multiselect(
            "Stack technique :",
            ["Git", "Docker", "Kubernetes", "AWS", "Azure", "GCP", "Flask", "Django", "FastAPI", "React", "Node.js", "MongoDB"],
            key="outils_dev"
        )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Design et Creativite**")
        outils_design = st.multiselect(
            "Suite creative :",
            ["Photoshop", "Illustrator", "InDesign", "Figma", "Canva", "After Effects", "Premiere Pro", "Sketch"],
            key="outils_design"
        )

    with col2:
        st.markdown("**Marketing et Communication**")
        outils_marketing = st.multiselect(
            "Outils marketing :",
            ["Google Analytics", "Meta Business Suite", "Google Ads", "SEMrush", "Mailchimp", "HubSpot", "Hootsuite", "Buffer"],
            key="outils_marketing"
        )

    st.divider()

    st.subheader("6. Parcours et experience")

    col1, col2 = st.columns(2)

    with col1:
        niveau_etudes = st.selectbox(
            "Niveau d'etudes :",
            ["Bac", "Bac+2 (BTS/DUT)", "Bac+3 (Licence)", "Bac+5 (Master)", "Bac+8 (Doctorat)", "Autre"],
            key="etudes"
        )
        domaine_formation = st.text_input(
            "Domaine de formation :",
            placeholder="Ex : Data Science, Marketing, Droit des affaires...",
            key="formation"
        )

    with col2:
        annees_experience = st.selectbox(
            "Annees d'experience professionnelle :",
            ["Étudiant / 0 an", "0-2 ans", "3-5 ans", "6-10 ans", "+10 ans"],
            key="experience"
        )
        secteur_actuel = st.text_input(
            "Secteur d'activite actuel ou recent :",
            placeholder="Ex : Banque, E-commerce, Conseil, Startup tech...",
            key="secteur"
        )

    st.divider()

    submit = st.form_submit_button("Analyser mon profil", width="stretch", type="primary")


if submit:

    likert_levels = {
        "business":      business_level,
        "finance":       finance_level,
        "design":        design_level,
        "communication": communication_level,
        "data_analysis": data_analysis_level,
        "ml":            ml_level,
        "dev":           dev_level,
        "engineering":   engineering_level,
    }

    juridique_level = st.session_state.get("_juridique_level", None)
    if juridique_level:
        likert_levels["juridique"] = juridique_level

    likert_summary = likert_to_semantic_text(likert_levels)

    soft_skills_summary = f"""
    Style de travail : {rigoeur}.
    Leadership : {leadership}.
    Persuasion : {persuasion}.
    Empathie : {empathie}.
    """

    outils_all = outils_data + outils_ml + outils_dev + outils_design + outils_marketing

    background_summary = f"""
    Formation : {niveau_etudes} en {domaine_formation}.
    Experience : {annees_experience} dans le secteur {secteur_actuel}.
    """

    domaines_text = f"Je m'intéresse aux domaines suivants : {', '.join(domaines)}." if domaines else ""
    outils_text   = f"Je maîtrise les outils et technologies suivants : {', '.join(outils_all)}." if outils_all else ""

    # injection B01 pour le profil Juriste
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
        domaines_text,
        juridique_boost,
        likert_summary,
        soft_skills_summary,
        projet_tech,
        journee_ideale,
        interet_specifique,
        defis_aimes,
        objectif_carriere,
        outils_text,
        background_summary
    ]).strip()

    if len(user_text.split()) < 30:
        st.warning("Reponses trop courtes. Veuillez detailler davantage vos experiences et aspirations pour une analyse fiable.")
    else:
        with st.spinner("Pretraitement semantique en cours..."):
            enriched_projet  = enrich_short_text(projet_tech)       if len(projet_tech.split()) < 15       else projet_tech
            enriched_interet = enrich_short_text(interet_specifique) if len(interet_specifique.split()) < 10 else interet_specifique
            final_text = user_text.replace(projet_tech, enriched_projet).replace(interet_specifique, enriched_interet)

        with st.spinner("Analyse semantique en cours..."):
            comp_scores = analyze_profile(
                final_text, bi_model, comp_ids, embeddings,
                None, comp_idx, top_k=None
            )
            top_jobs, bloc_scores = recommend_jobs(comp_scores, data, top_n=3)

        st.success("Analyse terminee.")

        st.divider()
        st.header("Resultats")

        col1, col2, col3 = st.columns(3)

        with col1:
            overall_score = top_jobs[0]['score'] if top_jobs else 0
            st.metric("Score de match", f"{overall_score:.0%}")

        with col2:
            nb_comp_fortes = sum(1 for b in top_jobs[0]['blocs'] if b['score'] >= 0.7)
            st.metric("Blocs maitrises", f"{nb_comp_fortes}/{len(top_jobs[0]['blocs'])}")

        with col3:
            potentiel = "Excellent" if overall_score >= 0.75 else "Tres bon" if overall_score >= 0.6 else "Bon"
            st.metric("Potentiel", potentiel)

        st.divider()

        st.subheader("Top 3 des metiers recommandes")

        tabs = st.tabs([f"#{i+1}  {job['titre']}  ({job['score']:.0%})" for i, job in enumerate(top_jobs)])

        for i, (tab, job) in enumerate(zip(tabs, top_jobs)):
            with tab:
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"### {job['titre']}")

                    global_score = float(job['score'])
                    st.progress(max(0.0, min(1.0, global_score)))

                    if job.get('description'):
                        st.info(f"**Description :** {job['description']}")

                    if job.get('blocs') and len(job['blocs']) > 0:
                        labels = [b['nom'] for b in job['blocs']]
                        values = [float(b['score']) for b in job['blocs']]

                        if len(labels) < 3:
                            fig = go.Figure(go.Bar(
                                x=values,
                                y=labels,
                                orientation='h',
                                marker=dict(color='#1f77b4', line=dict(color='white', width=1)),
                                text=[f"{v:.0%}" for v in values],
                                textposition='auto',
                            ))
                            fig.update_layout(
                                title=f"Adequation par bloc - {job['titre']}",
                                xaxis=dict(range=[0, 1], tickformat='.0%'),
                                yaxis=dict(autorange="reversed"),
                                height=300,
                                margin=dict(l=20, r=20, t=50, b=20),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)'
                            )
                        else:
                            labels_radar = labels + [labels[0]]
                            values_radar = values + [values[0]]

                            fig = go.Figure()
                            fig.add_trace(go.Scatterpolar(
                                r=values_radar,
                                theta=labels_radar,
                                fill='toself',
                                fillcolor='rgba(31, 119, 180, 0.3)',
                                name='Votre profil',
                                line=dict(color='#1f77b4', width=3),
                                marker=dict(size=8)
                            ))
                            fig.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                        range=[0, 1],
                                        tickformat='.0%',
                                        gridcolor="#444",
                                        tickfont=dict(size=10)
                                    ),
                                    angularaxis=dict(
                                        gridcolor="#444",
                                        tickfont=dict(size=11)
                                    ),
                                    bgcolor='rgba(0,0,0,0)'
                                ),
                                showlegend=False,
                                height=450,
                                margin=dict(l=80, r=80, t=60, b=40),
                                title=f"Cartographie des competences - {job['titre']}"
                            )

                        st.plotly_chart(fig, width="stretch")

                with col2:
                    st.metric("Score global", f"{global_score:.0%}")

                    if i < len(top_jobs) - 1:
                        delta = global_score - float(top_jobs[i+1]['score'])
                        st.metric(f"Ecart avec #{i+2}", f"+{delta:.1%}")

                    st.write("---")
                    with st.expander("Detail des blocs", expanded=True):
                        for bloc in job['blocs']:
                            b_score = float(bloc['score'])
                            st.write(f"**{bloc['nom']}**")
                            st.progress(max(0.0, min(1.0, b_score)))
                            st.caption(f"Adequation : {b_score:.0%}")

        st.divider()
        st.header("Recommandations")

        col1, col2 = st.columns([3, 2])

        with col1:
            st.subheader("Plan de progression")

            weak_blocs = {
                b['nom']: b['score']
                for b in top_jobs[0]['blocs']
                if b['score'] < 0.6
            }

            if weak_blocs:
                with st.spinner("Generation du plan personnalise..."):
                    learning_path = generate_learning_path(
                        weak_blocs,
                        top_jobs[0]['titre'],
                        user_text[:600]
                    )
                st.markdown(learning_path)
            else:
                st.success("Vous maitrisez deja tous les blocs requis pour ce metier.")

        with col2:
            st.subheader("Synthese de profil")

            strong_blocs = {
                b['nom']: b['score']
                for b in top_jobs[0]['blocs']
                if b['score'] >= 0.6
            }

            with st.spinner("Redaction de la synthese..."):
                bio = generate_professional_bio(
                    strong_blocs,
                    top_jobs[0]['titre'],
                    {
                        "projet_tech":    projet_tech,
                        "journee_ideale": journee_ideale,
                        "objectif":       objectif_carriere
                    }
                )

            st.info(bio)
            st.download_button(
                "Telecharger ma synthese",
                bio,
                file_name="synthese_aisca.txt",
                mime="text/plain",
                width="stretch"
            )

        st.divider()
        st.subheader("Tableau recapitulatif")

        df_scores = pd.DataFrame([
            {
                "Bloc de competences": b['nom'],
                "Score": f"{b['score']:.0%}",
                "Niveau": "Maitrise" if b['score'] >= 0.7 else "Intermediaire" if b['score'] >= 0.4 else "A developper"
            }
            for b in top_jobs[0]['blocs']
        ]).sort_values("Score", ascending=False)

        st.dataframe(df_scores, width="stretch", hide_index=True)

        st.balloons()

        # debug
        with st.expander("DEBUG - Texte utilisateur genere", expanded=False):
            st.text_area("Texte complet envoye a l'analyse :", user_text, height=200)
            st.write(f"**Mots :** {len(user_text.split())}  |  **Caracteres :** {len(user_text)}")
            if juridique_boost:
                st.success(f"Boost juridique actif : {len(juridique_boost.split())} mots injectes (B01 x repetitions)")
            else:
                st.info("Pas de boost juridique.")

        with st.expander("DEBUG - Likert vers texte semantique", expanded=False):
            st.text_area("Texte semantique genere depuis les sliders :", likert_summary, height=150)

        with st.expander("DEBUG - Texte apres enrichissement GenAI", expanded=False):
            if enriched_projet != projet_tech:
                st.success(f"Projet enrichi : {len(enriched_projet.split())} mots (original : {len(projet_tech.split())})")
            if enriched_interet != interet_specifique:
                st.success(f"Interet enrichi : {len(enriched_interet.split())} mots (original : {len(interet_specifique.split())})")
            st.text_area("Texte final envoye a l'analyse NLP :", final_text, height=200)

        with st.expander("DEBUG - Scores des competences detectees", expanded=False):
            if comp_scores:
                top_comps = sorted(comp_scores.items(), key=lambda x: -x[1])[:10]
                st.write("**Top 10 competences matchees :**")
                for cid, score in top_comps:
                    st.write(f"- {comp_idx[cid]['texte']} : {score:.2%}")
            else:
                st.error("Aucune competence detectee.")

st.divider()
st.caption("OSCC - 2026 | EFREI - Master Data Engineering & AI | Projet IA Generative")
