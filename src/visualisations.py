"""
Interface Streamlit - AISCA v2.0
Questionnaire adapté au référentiel de 11 blocs et 40 métiers
"""
# streamlit run src/visualisations.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
#from streamlit_extras.let_it_rain import rain

# ✅ CORRECTION : Noms de fonctions corrects (snake_case)
from scoring import get_models, load_and_index_data, analyze_profile, recommend_jobs
from genai_augmentation import enrich_short_text, generate_learning_path, generate_professional_bio


# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="AISCA - Orientation IA",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_PATH = "data/referentiel.json"


# ============================================================================
# INITIALISATION
# ============================================================================

@st.cache_resource
def initialize_app():
    try:
        bi_model, cross_model = get_models()
        data, comp_idx, comp_ids, embeddings = load_and_index_data(DATA_PATH)
        return bi_model, cross_model, data, comp_idx, comp_ids, embeddings
    except Exception as e:
        st.error(f"❌ Erreur d'initialisation : {e}")
        st.stop()


with st.spinner("🔄 Chargement des modèles NLP..."):
    bi_model, cross_model, data, comp_idx, comp_ids, embeddings = initialize_app()


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.image("src/visuel/img/projet_IA_Generative.png", use_container_width=True)
    st.header("ℹ️ Agent RAG Intelligent")
    st.markdown("""
    **Pipeline d'analyse :**
    1. 🔍 **Retrieval** : Matching sémantique SBERT
    2. 🧠 **Augmentation** : Enrichissement GenAI
    3. 📊 **Generation** : Recommandations personnalisées
    
    **11 blocs de compétences**  
    **40 métiers analysés**  
    **200+ compétences indexées**
    """)

    st.divider()
    st.caption("**Stack technique :**")
    st.code("NLP: SBERT + Cross-Encoder\nGenAI: Gemini 2.5 Flash", language="text")
with st.sidebar:
    st.divider()
    st.header("🧪 Test Rapide")
    if st.button("Remplir un profil de test", use_container_width=True):
        st.session_state.update({
            "business": "Intermédiaire",
            "finance": "Débutant",
            "design": "Débutant",
            "communication": "Intermédiaire",
            "data_analysis": "Avancé",
            "ml": "Avancé",
            "dev": "Intermédiaire",
            "engineering": "Débutant",
            "rigueur": "Rigoureux et attentif aux détails",
            "leadership": "Je prends naturellement le lead",
            "persuasion": "Parfois, selon le sujet",
            "empathie": "Je trouve un équilibre",
            "projet": "J'ai développé un système de prédiction de churn client avec Python, scikit-learn et XGBoost atteignant 92% de précision",
            "journee": "Alterner analyse de données, création de modèles ML et présentation des résultats à l'équipe",
            "interet": "Machine Learning, Data Science, Intelligence Artificielle",
            "defis": "Optimiser des modèles prédictifs et trouver des insights dans des datasets complexes",
            "objectif": "Devenir Lead Data Scientist dans une fintech",
            "outils_data": ["Python", "Pandas", "SQL", "Matplotlib"],
            "outils_ml": ["Scikit-learn", "TensorFlow", "PyTorch"],
            "outils_dev": ["Git", "Docker"],
            "formation": "Data Science",
            "etudes": "Bac+5 (Master)",
            "experience": "3-5 ans",
            "secteur": "Fintech"
        })
        st.rerun()


# ============================================================================
# INTERFACE PRINCIPALE
# ============================================================================

st.title("🎯 AISCA - Agent Intelligent de Cartographie des Compétences")
st.markdown("*Analyse sémantique avancée pour l'orientation professionnelle*")

st.divider()

# ============================================================================
# QUESTIONNAIRE HYBRIDE
# ============================================================================

st.header("📝 Questionnaire d'Évaluation des Compétences")
st.markdown("Répondez aux questions suivantes pour obtenir votre profil personnalisé.")

with st.form(key="questionnaire_complet"):

    # === SECTION 1 : DOMAINES D'INTÉRÊT ===
    st.subheader("1️⃣ Vos domaines d'intérêt professionnels")
    st.caption("Sélectionnez tous les domaines qui vous attirent")

    col1, col2, col3 = st.columns(3)

    with col1:
        domaines_col1 = st.multiselect(
            "Domaines business & juridique :",
            ["Juridique", "Business et Stratégie", "Marketing et Vente",
             "Finance et Comptabilité"],
            help="Compétences en droit, stratégie, commerce et finance"
        )

    with col2:
        domaines_col2 = st.multiselect(
            "Domaines créatifs & communication :",
            ["Communication et Médias", "Création et Design",
             "Digital et Réseaux Sociaux"],
            help="Compétences en communication, design et social media"
        )

    with col3:
        domaines_col3 = st.multiselect(
            "Domaines techniques & data :",
            ["Data Analysis", "Machine Learning et IA",
             "Développement et Infrastructure", "Ingénierie et Technique"],
            help="Compétences en data, IA, développement et ingénierie"
        )

    domaines = domaines_col1 + domaines_col2 + domaines_col3

    st.divider()

    # === SECTION 2 : AUTO-ÉVALUATION (Likert) ===
    st.subheader("2️⃣ Auto-évaluation de vos compétences techniques")
    st.caption("Évaluez honnêtement votre niveau actuel dans chaque domaine")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**💼 Business & Stratégie**")
        business_level = st.select_slider(
            "Stratégie d'entreprise",
            options=["Débutant", "Intermédiaire", "Avancé", "Expert"],
            value="Débutant",
            key="business"
        )

        finance_level = st.select_slider(
            "Finance / Comptabilité",
            options=["Débutant", "Intermédiaire", "Avancé", "Expert"],
            value="Débutant",
            key="finance"
        )

    with col2:
        st.markdown("**🎨 Créativité & Communication**")
        design_level = st.select_slider(
            "Design / Créativité",
            options=["Débutant", "Intermédiaire", "Avancé", "Expert"],
            value="Débutant",
            key="design"
        )

        communication_level = st.select_slider(
            "Communication / Média",
            options=["Débutant", "Intermédiaire", "Avancé", "Expert"],
            value="Débutant",
            key="communication"
        )

    with col3:
        st.markdown("**📊 Data & Analyse**")
        data_analysis_level = st.select_slider(
            "Analyse de données",
            options=["Débutant", "Intermédiaire", "Avancé", "Expert"],
            value="Débutant",
            key="data_analysis"
        )

        ml_level = st.select_slider(
            "Machine Learning / IA",
            options=["Débutant", "Intermédiaire", "Avancé", "Expert"],
            value="Débutant",
            key="ml"
        )

    with col4:
        st.markdown("**💻 Développement & Tech**")
        dev_level = st.select_slider(
            "Développement logiciel",
            options=["Débutant", "Intermédiaire", "Avancé", "Expert"],
            value="Débutant",
            key="dev"
        )

        engineering_level = st.select_slider(
            "Ingénierie technique",
            options=["Débutant", "Intermédiaire", "Avancé", "Expert"],
            value="Débutant",
            key="engineering"
        )

    st.divider()

    # === SECTION 3 : SOFT SKILLS ===
    st.subheader("3️⃣ Vos traits de personnalité et soft skills")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🧠 Style de travail**")

        rigoeur = st.radio(
            "Vous êtes plutôt :",
            ["Rigoureux et attentif aux détails",
             "Créatif et improvisateur",
             "Équilibré entre les deux"],
            key="rigueur"
        )

        leadership = st.radio(
            "Face à un projet d'équipe :",
            ["Je prends naturellement le lead",
             "Je préfère contribuer en tant que membre",
             "Cela dépend du contexte"],
            key="leadership"
        )

    with col2:
        st.markdown("**💡 Compétences relationnelles**")

        persuasion = st.radio(
            "Aimez-vous débattre et convaincre ?",
            ["Oui, j'adore argumenter et persuader",
             "Non, je préfère éviter les confrontations",
             "Parfois, selon le sujet"],
            key="persuasion"
        )

        empathie = st.radio(
            "Dans vos relations professionnelles :",
            ["Je suis très empathique et à l'écoute",
             "Je privilégie l'efficacité sur l'émotion",
             "Je trouve un équilibre"],
            key="empathie"
        )

    st.divider()

    # === SECTION 4 : QUESTIONS OUVERTES ===
    st.subheader("4️⃣ Vos expériences et aspirations")
    st.caption("Détaillez vos réponses pour une analyse plus précise")

    projet_tech = st.text_area(
        "🚀 Décrivez un projet technique ou professionnel dont vous êtes fier :",
        placeholder="Ex : J'ai développé un dashboard Power BI pour analyser les ventes en temps réel...",
        height=120,
        key="projet"
    )

    col1, col2 = st.columns(2)

    with col1:
        journee_ideale = st.text_area(
            "☀️ Décrivez votre journée de travail idéale :",
            placeholder="Ex : Alterner entre analyse de données, réunions stratégiques...",
            height=100,
            key="journee"
        )

        interet_specifique = st.text_area(
            "💼 Domaines spécifiques qui vous passionnent :",
            placeholder="Ex : L'éthique de l'IA, l'optimisation énergétique...",
            height=100,
            key="interet"
        )

    with col2:
        defis_aimes = st.text_area(
            "💡 Types de défis qui vous stimulent :",
            placeholder="Ex : Résoudre des problèmes complexes avec la data...",
            height=100,
            key="defis"
        )

        objectif_carriere = st.text_area(
            "🎯 Objectif de carrière à 3-5 ans :",
            placeholder="Ex : Devenir Lead Data Scientist dans une fintech...",
            height=100,
            key="objectif"
        )

    st.divider()

    # === SECTION 5 : OUTILS & TECHNOLOGIES ===
    st.subheader("5️⃣ Outils et technologies maîtrisés")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📊 Data & Analytics**")
        outils_data = st.multiselect(
            "Outils maîtrisés :",
            ["Python", "R", "SQL", "Excel avancé", "Pandas", "NumPy",
             "Matplotlib", "Seaborn", "Plotly", "Tableau", "Power BI"],
            key="outils_data"
        )

    with col2:
        st.markdown("**🤖 IA & Machine Learning**")
        outils_ml = st.multiselect(
            "Frameworks IA :",
            ["Scikit-learn", "TensorFlow", "PyTorch", "Keras",
             "HuggingFace", "LangChain", "OpenAI API", "BERT", "GPT"],
            key="outils_ml"
        )

    with col3:
        st.markdown("**💻 Développement & Cloud**")
        outils_dev = st.multiselect(
            "Stack technique :",
            ["Git", "Docker", "Kubernetes", "AWS", "Azure", "GCP",
             "Flask", "Django", "FastAPI", "React", "Node.js", "MongoDB"],
            key="outils_dev"
        )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🎨 Design & Créativité**")
        outils_design = st.multiselect(
            "Suite créative :",
            ["Photoshop", "Illustrator", "InDesign", "Figma", "Canva",
             "After Effects", "Premiere Pro", "Sketch"],
            key="outils_design"
        )

    with col2:
        st.markdown("**📱 Marketing & Communication**")
        outils_marketing = st.multiselect(
            "Outils marketing :",
            ["Google Analytics", "Meta Business Suite", "Google Ads",
             "SEMrush", "Mailchimp", "HubSpot", "Hootsuite", "Buffer"],
            key="outils_marketing"
        )

    st.divider()

    # === SECTION 6 : EXPÉRIENCE & FORMATION ===
    st.subheader("6️⃣ Parcours et expérience")

    col1, col2 = st.columns(2)

    with col1:
        niveau_etudes = st.selectbox(
            "Niveau d'études :",
            ["Bac", "Bac+2 (BTS/DUT)", "Bac+3 (Licence)",
             "Bac+5 (Master)", "Bac+8 (Doctorat)", "Autre"],
            key="etudes"
        )

        domaine_formation = st.text_input(
            "Domaine de formation :",
            placeholder="Ex : Data Science, Marketing, Droit des affaires...",
            key="formation"
        )

    with col2:
        annees_experience = st.selectbox(
            "Années d'expérience professionnelle :",
            ["Étudiant / 0 an", "0-2 ans", "3-5 ans", "6-10 ans", "+10 ans"],
            key="experience"
        )

        secteur_actuel = st.text_input(
            "Secteur d'activité actuel/récent :",
            placeholder="Ex : Banque, E-commerce, Conseil, Startup tech...",
            key="secteur"
        )

    st.divider()

    # === SUBMISSION ===
    submit = st.form_submit_button("🚀 Analyser mon profil complet", use_container_width=True, type="primary")


# ============================================================================
# TRAITEMENT & ANALYSE
# ============================================================================

if submit:
    # Construction du texte utilisateur enrichi
    likert_summary = f"""
    Niveaux de compétences : 
    Business/Stratégie {business_level}, Finance {finance_level}, 
    Design {design_level}, Communication {communication_level},
    Data Analysis {data_analysis_level}, Machine Learning {ml_level},
    Développement {dev_level}, Ingénierie {engineering_level}.
    """

    soft_skills_summary = f"""
    Style de travail : {rigoeur}.
    Leadership : {leadership}.
    Persuasion : {persuasion}.
    Empathie : {empathie}.
    """

    outils_all = outils_data + outils_ml + outils_dev + outils_design + outils_marketing

    background_summary = f"""
    Formation : {niveau_etudes} en {domaine_formation}.
    Expérience : {annees_experience} dans le secteur {secteur_actuel}.
    """

    user_text = " ".join([
        " ".join(domaines),
        likert_summary,
        soft_skills_summary,
        projet_tech,
        journee_ideale,
        interet_specifique,
        defis_aimes,
        objectif_carriere,
        " ".join(outils_all),
        background_summary
    ]).strip()

    if len(user_text.split()) < 30:
        st.warning("⚠️ Réponses trop courtes. Veuillez détailler davantage vos expériences et aspirations pour une analyse fiable.")
    else:
        # === ÉTAPE 1 : Enrichissement GenAI conditionnel ===
        with st.spinner("🔍 Prétraitement sémantique avec GenAI..."):
            enriched_projet = enrich_short_text(projet_tech) if len(projet_tech.split()) < 15 else projet_tech
            enriched_interet = enrich_short_text(interet_specifique) if len(interet_specifique.split()) < 10 else interet_specifique

            final_text = user_text.replace(projet_tech, enriched_projet).replace(interet_specifique, enriched_interet)

        # === ÉTAPE 2 : Analyse sémantique (RAG - Retrieval) ===
        with st.spinner("🧠 Analyse NLP sémantique en cours..."):
            comp_scores = analyze_profile(
                final_text, bi_model, comp_ids, embeddings,
                cross_model, comp_idx, top_k=40
            )

            top_jobs, bloc_scores = recommend_jobs(comp_scores, data, top_n=3)

        st.success("✅ Analyse terminée avec succès !")

        # === AFFICHAGE DES RÉSULTATS ===
        st.divider()
        st.header("🏆 Vos Résultats Personnalisés")

        # Métrique globale
        col1, col2, col3 = st.columns(3)

        with col1:
            overall_score = top_jobs[0]['score'] if top_jobs else 0
            st.metric("Score de Match #1", f"{overall_score:.0%}")

        with col2:
            nb_comp_fortes = sum(1 for b in top_jobs[0]['blocs'] if b['score'] >= 0.7)
            st.metric("Blocs Maîtrisés", f"{nb_comp_fortes}/{len(top_jobs[0]['blocs'])}")

        with col3:
            potentiel = "Excellent" if overall_score >= 0.75 else "Très bon" if overall_score >= 0.6 else "Bon"
            st.metric("Potentiel", potentiel)

        st.divider()

        # === TABS DES MÉTIERS RECOMMANDÉS ===
        st.subheader("🎯 Top 3 des métiers recommandés")

        tabs = st.tabs([f"#{i+1} {job['titre']} ({job['score']:.0%})" for i, job in enumerate(top_jobs)])

        for i, (tab, job) in enumerate(zip(tabs, top_jobs)):
            with tab:
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"### {job['titre']}")
                    st.progress(job['score'])

                    if job.get('description'):
                        st.info(f"**Description :** {job['description']}")

                    # Radar chart des blocs de compétences
                    if job['blocs']:
                        fig = go.Figure()

                        fig.add_trace(go.Scatterpolar(
                            r=[b['score'] for b in job['blocs']],
                            theta=[b['nom'][:20] for b in job['blocs']],
                            fill='toself',
                            name='Votre profil',
                            line=dict(color='#1f77b4', width=2)
                        ))

                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1],
                                    tickformat='.0%'
                                )
                            ),
                            showlegend=False,
                            height=400,
                            title=f"Cartographie des compétences pour {job['titre']}"
                        )

                        st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.metric("Score Global", f"{job['score']:.0%}")

                    if i < len(top_jobs) - 1:
                        delta = job['score'] - top_jobs[i+1]['score']
                        st.metric("Écart avec #" + str(i+2), f"+{delta:.1%}")

                    with st.expander("📊 Détail des blocs"):
                        for bloc in job['blocs']:
                            st.write(f"**{bloc['nom']}**")
                            st.progress(bloc['score'])
                            st.caption(f"Score : {bloc['score']:.0%}")
                            st.divider()

        # === GÉNÉRATION PLAN + BIO (RAG - Generation) ===
        st.divider()
        st.header("🎯 Recommandations Stratégiques")

        col1, col2 = st.columns([3, 2])

        with col1:
            st.subheader("📚 Plan de Progression Personnalisé")

            weak_blocs = {
                b['nom']: b['score']
                for b in top_jobs[0]['blocs']
                if b['score'] < 0.6
            }

            if weak_blocs:
                with st.spinner("🤖 Génération du plan avec IA générative..."):
                    learning_path = generate_learning_path(
                        weak_blocs,
                        top_jobs[0]['titre'],
                        user_text[:600]
                    )

                st.markdown(learning_path)
            else:
                st.success("🎉 Félicitations ! Vous maîtrisez déjà tous les blocs requis pour ce métier.")

        with col2:
            st.subheader("💼 Votre Bio Professionnelle")

            strong_blocs = {
                b['nom']: b['score']
                for b in top_jobs[0]['blocs']
                if b['score'] >= 0.6
            }

            with st.spinner("✍️ Rédaction de votre bio..."):
                bio = generate_professional_bio(
                    strong_blocs,
                    top_jobs[0]['titre'],
                    {
                        "projet_tech": projet_tech,
                        "journee_ideale": journee_ideale,
                        "objectif": objectif_carriere
                    }
                )

            st.info(bio)

            st.download_button(
                "📥 Télécharger ma bio",
                bio,
                file_name="bio_professionnelle_aisca.txt",
                mime="text/plain",
                use_container_width=True
            )

        # === TABLEAU RÉCAPITULATIF DES SCORES PAR BLOC ===
        st.divider()
        st.subheader("📊 Tableau récapitulatif de vos scores")

        df_scores = pd.DataFrame([
            {"Bloc de compétences": b['nom'], "Score": f"{b['score']:.0%}", "Niveau":
             "🟢 Maîtrisé" if b['score'] >= 0.7 else "🟡 Intermédiaire" if b['score'] >= 0.4 else "🔴 À développer"}
            for b in top_jobs[0]['blocs']
        ]).sort_values("Score", ascending=False)

        st.dataframe(df_scores, use_container_width=True, hide_index=True)

        # Affichage de ballon
        st.balloons()
        """
        Partie DEBUG
        """
        # ✅ DEBUG : Afficher le texte généré
        with st.expander("🔍 DEBUG : Texte utilisateur généré", expanded=False):
            st.text_area("Texte complet envoyé à l'analyse :", user_text, height=200)
            st.write(f"**Nombre de mots :** {len(user_text.split())}")
            st.write(f"**Nombre de caractères :** {len(user_text)}")

        # ✅ DEBUG : Afficher le texte enrichi
        with st.expander("🔍 DEBUG : Texte après enrichissement GenAI", expanded=False):
            if enriched_projet != projet_tech:
                st.success(
                    f"✅ Projet enrichi : {len(enriched_projet.split())} mots (original : {len(projet_tech.split())})")
            if enriched_interet != interet_specifique:
                st.success(
                    f"✅ Intérêt enrichi : {len(enriched_interet.split())} mots (original : {len(interet_specifique.split())})")
            st.text_area("Texte final envoyé à l'analyse NLP :", final_text, height=200)

        # ✅ DEBUG : Afficher les scores de compétences
        with st.expander("🔍 DEBUG : Scores des compétences détectées", expanded=False):
            if comp_scores:
                top_comps = sorted(comp_scores.items(), key=lambda x: -x[1])[:10]
                st.write("**Top 10 compétences matchées :**")
                for cid, score in top_comps:
                    st.write(f"- {comp_idx[cid]['texte']} : {score:.2%}")
            else:
                st.error("❌ Aucune compétence détectée ! Problème dans l'analyse.")

st.divider()
st.caption("AISCA © 2026 | EFREI - Master Data Engineering & AI | Projet IA Générative - Bloc RNCP40875")
