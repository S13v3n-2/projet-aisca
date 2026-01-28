import streamlit as st
import pandas as pd
import numpy as np
from scoring import call_model

st.title('Trouvez le métier fait pour vous.')
with st.form(key="formulaire", clear_on_submit=True, enter_to_submit=True, border=True, width="stretch", height="content"):
    Categorie = st.multiselect("Domaine qui vous interesse",["Juridique","Business","Stratégie","Marketing","Finance","Comptabilité",
                                           "Communication","Design","Digital et Réseaux Sociaux","Data Analysis",
                                           "Machine Learning","Intelligence Artificielle","Développement","Ingénierie Technique"])

    Experience_passe = st.text_area(
        "Décrivez un projet technique dont vous êtes fier. Quelles technologies avez-vous utilisées et quels problèmes avez-vous résolus ?",
    )

    st.write(f"You wrote {len(Experience_passe)} characters.")

    Mission_quotidienne = st.text_area(
        "Si vous deviez décrire votre journée de travail idéale en termes de tâches techniques, que diriez-vous ?",
    )

    st.write(f"You wrote {len(Mission_quotidienne)} characters.")

    Negociation = st.text_area(
        "Aimez-vous débattre, argumenter et convaincre des interlocuteurs dans un cadre professionnel ?",
    )

    st.write(f"You wrote {len(Mission_quotidienne)} characters.")
    Rigueur = st.text_area(
        "Préférez-vous un travail exigeant une précision millimétrée (chiffres, clauses) ou un travail laissant place à l'improvisation créative ?",
    )

    st.write(f"You wrote {len(Mission_quotidienne)} characters.")

    st.form_submit_button(label="Submit")

if Rigueur:
    top_metiers, scores_blocs = call_model(Rigueur)
    for index, metier in enumerate(top_metiers[:3]):
        print(f"Top {index + 1} Métier : {metier['titre']} ({metier['score']:.2%})")
        st.write(f"Top {index + 1} Métier : {metier['titre']} ({metier['score']:.2%})")
st.write(Categorie)
st.write(Rigueur)