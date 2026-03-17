# module d'augmentation par IA générative avec Gemini
# on l'utilise pour enrichir les textes trop courts et générer les plans de progression
# on a mis un cache fichier pour éviter de cramer le free tier de l'api

import os
import json
import hashlib
from typing import Optional, Dict
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

# on charge la clé api depuis le .env qui est dans le même dossier que le script
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# on crash plus si la clé est pas là, comme ça on peut au moins lancer l'interface
# les fonctions gemini renverront juste un message d'erreur
if not GEMINI_API_KEY:
    print("GEMINI_API_KEY introuvable dans .env - les fonctions Gemini seront desactivees")

# dossier de cache pour stocker les réponses de l'api
# comme ça si on relance avec les mêmes inputs on tape pas l'api
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# on utilise gemini-2.5-flash, c'est le modèle stable avec 1M tokens de contexte
# on avait testé gemini-pro avant mais flash est plus rapide et suffisant pour notre usage
MODEL_NAME = "gemini-2.5-flash"

generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 4096,
}

# on désactive tous les filtres de sécurité parce que sinon ça bloquait
# des réponses parfaitement normales sur les compétences professionnelles
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# seuils pour l'enrichissement de texte
# en dessous de 15 mots on considère que c'est trop court
# en dessous de 5 mots on enrichit systématiquement
MAX_SHORT_TEXT_LENGTH = 15
MIN_ENRICHMENT_LENGTH = 5


# on hash le prompt pour avoir une clé unique par requête
def get_cache_key(prompt: str, prefix: str = "") -> str:
    content = f"{prefix}:{prompt}"
    return hashlib.sha256(content.encode()).hexdigest()


# charge une réponse depuis le cache si elle existe
def load_from_cache(cache_key: str) -> Optional[str]:
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("response")
        except (json.JSONDecodeError, IOError) as e:
            print(f"Erreur lecture cache : {e}")
            return None
    return None


# sauvegarde une réponse dans le cache
def save_to_cache(cache_key: str, response: str):
    cache_file = CACHE_DIR / f"{cache_key}.json"
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({"response": response}, f, ensure_ascii=False, indent=2)
    except IOError as e:
        print(f"Erreur écriture cache : {e}")


# fonction générique pour appeler gemini avec gestion du cache et des erreurs
# on passe par là pour tous les appels api histoire de centraliser
def call_gemini_api(prompt: str, cache_prefix: str) -> str:
    # si y'a pas de clé api on retourne direct un message d'erreur
    if not GEMINI_API_KEY:
        return "Erreur API Gemini : cle API manquante, cree un .env avec GEMINI_API_KEY=ta_cle"

    cache_key = get_cache_key(prompt, prefix=cache_prefix)
    cached = load_from_cache(cache_key)

    if cached:
        print(f"Cache hit pour {cache_prefix}")
        return cached

    try:
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        print(f"Appel API Gemini ({cache_prefix})...")
        response = model.generate_content(prompt)

        if not response or not response.text:
            if hasattr(response, 'prompt_feedback'):
                return f"Reponse bloquee : {response.prompt_feedback}"
            return "Pas de reponse generee."

        generated_text = response.text.strip()
        save_to_cache(cache_key, generated_text)
        print(f"Reponse generee et mise en cache")

        return generated_text

    except Exception as e:
        error_msg = f"Erreur API Gemini : {type(e).__name__} - {str(e)}"
        print(error_msg)
        return error_msg


# enrichit un texte trop court via gemini
# on s'en sert quand l'utilisateur écrit genre "python data" et c'est tout
# gemini reformule en ajoutant du contexte pro sans inventer des compétences
def enrich_short_text(user_text: str) -> str:
    if not user_text or not user_text.strip():
        return user_text

    word_count = len(user_text.split())

    if word_count >= MIN_ENRICHMENT_LENGTH:
        print(f"Texte suffisamment long ({word_count} mots), pas d'enrichissement")
        return user_text

    print(f"Enrichissement texte court ({word_count} mots)...")

    prompt = f"""Tu es un expert en orientation professionnelle.
L'utilisateur a donné une réponse très courte : "{user_text}"

Réécris cette phrase en ajoutant du contexte technique professionnel pertinent,
sans inventer de compétences non mentionnées. Maximum 3 phrases.

Réponse enrichie (uniquement le texte, pas de préambule) :"""

    enriched = call_gemini_api(prompt, cache_prefix="enrich")

    if enriched.startswith("Erreur") or enriched.startswith("Reponse") or enriched.startswith("Pas de"):
        print(f"Echec enrichissement, texte original conserve")
        return user_text

    return enriched


# genere un plan de progression personnalise et detaille
# on passe maintenant tout le profil utilisateur (niveaux, outils, textes, formation)
# pour que gemini puisse faire une vraie analyse croisee quanti + semantique
def generate_learning_path(
    weak_skills: Dict[str, float],
    strong_skills: Dict[str, float],
    target_job: str,
    user_profile_data: Dict
) -> str:
    if not weak_skills:
        return "Vous maitrisez deja tous les blocs requis pour ce metier."

    # --- construction du contexte complet pour gemini ---

    # 1. donnees quantitatives : niveaux likert auto-evalues
    likert = user_profile_data.get("likert_levels", {})
    likert_section = ""
    if likert:
        likert_section = "**Auto-évaluation technique (niveaux déclarés par l'utilisateur) :**\n"
        for domaine, niveau in likert.items():
            likert_section += f"- {domaine} : {niveau}\n"

    # 2. scores NLP par bloc (analyse semantique du modele)
    scores_section = "**Scores NLP calculés par notre modèle (similarité sémantique) :**\n"
    scores_section += "Blocs à développer :\n"
    for bloc, score in sorted(weak_skills.items(), key=lambda x: x[1]):
        scores_section += f"- {bloc} : {score:.0%}\n"
    if strong_skills:
        scores_section += "Blocs déjà maîtrisés :\n"
        for bloc, score in sorted(strong_skills.items(), key=lambda x: -x[1]):
            scores_section += f"- {bloc} : {score:.0%}\n"

    # 3. outils deja maitrises par l'utilisateur
    outils = user_profile_data.get("outils", {})
    outils_section = ""
    if outils and any(outils.values()):
        outils_section = "**Outils et technologies déjà maîtrisés :**\n"
        for cat, tools in outils.items():
            if tools:
                outils_section += f"- {cat} : {', '.join(tools)}\n"

    # 4. soft skills
    soft = user_profile_data.get("soft_skills", {})
    soft_section = ""
    if soft and any(soft.values()):
        soft_section = "**Soft skills déclarés :**\n"
        for trait, val in soft.items():
            if val:
                soft_section += f"- {trait} : {val}\n"

    # 5. textes libres (ce que l'etudiant a ecrit lui-meme)
    textes = user_profile_data.get("textes_libres", {})
    textes_section = ""
    if textes and any(textes.values()):
        textes_section = "**Réponses libres de l'utilisateur (à analyser sémantiquement) :**\n"
        for champ, txt in textes.items():
            if txt and txt.strip():
                textes_section += f"- {champ} : {txt[:300]}\n"

    # 6. parcours academique et pro
    formation = user_profile_data.get("formation", {})
    formation_section = ""
    if formation and any(formation.values()):
        formation_section = "**Parcours académique et professionnel :**\n"
        for k, v in formation.items():
            if v:
                formation_section += f"- {k} : {v}\n"

    context = f"**Métier cible :** {target_job}\n\n"
    context += "\n".join(filter(None, [
        likert_section, scores_section, outils_section,
        soft_section, textes_section, formation_section
    ]))

    prompt = f"""{context}

Tu es un conseiller expert en orientation professionnelle spécialisé Data/IA/Business/Droit/Design/Communication.
L'utilisateur est un étudiant ou jeune professionnel. Tu dois produire un plan de progression **ultra-personnalisé** en croisant :

1. **L'analyse quantitative** : les niveaux Likert auto-évalués ET les scores NLP calculés par notre modèle.
   Compare les deux : si l'utilisateur se déclare "Avancé" en Data Analysis mais que le score NLP est bas,
   signale cet écart et propose d'approfondir. Inversement, valorise les points forts confirmés.

2. **L'analyse sémantique** : les réponses libres de l'utilisateur (projet, journée idéale, intérêts, défis, objectif).
   Identifie ses motivations profondes, ses centres d'intérêt réels, son style de travail préféré.
   Adapte tes recommandations à CE profil précis, pas des conseils génériques.

3. **Les outils déclarés** : l'utilisateur maîtrise déjà certains outils. Ne recommande PAS ce qu'il sait déjà.
   Propose les outils COMPLÉMENTAIRES qu'il doit apprendre pour le métier {target_job}.

4. **Le niveau d'études et l'expérience** : adapte la difficulté et le type de ressources.
   Un Bac+5 avec 3-5 ans d'expérience ne reçoit pas les mêmes recommandations qu'un étudiant en Bac+3.

Génère le plan suivant en Markdown strict :

## Diagnostic personnalisé
Résumé en 3-4 phrases : points forts confirmés, écarts identifiés entre auto-évaluation et scores NLP,
et axes de progression prioritaires. Mentionne les soft skills de l'utilisateur et comment ils servent
(ou desservent) le métier cible.

## Plan de progression détaillé
Pour CHAQUE bloc à développer (du plus faible au plus fort), crée une sous-section :

### [Nom du bloc] — Score actuel : X%
- **Pourquoi c'est prioritaire** : explique en 1-2 phrases pourquoi ce bloc est critique pour {target_job}
- **Ce que l'utilisateur a déjà** : mentionne les outils/compétences qu'il possède déjà dans ce domaine
- **Ce qu'il doit apprendre** :
  - Compétence 1 → Cours précis (nom + plateforme : Coursera, Udemy, OpenClassrooms, DataCamp, etc.)
  - Compétence 2 → Cours précis
- **Projet pratique** : un projet concret à réaliser, en lien avec ses intérêts déclarés si possible
- **Certification recommandée** : uniquement si elle existe et est reconnue (Google, AWS, Microsoft, HubSpot, etc.).
  Précise le nom exact, le coût approximatif, et pourquoi elle est pertinente. Si aucune certification
  n'est vraiment adaptée, ne force pas — dis-le clairement.

## Outils à acquérir
Liste les outils/technologies que l'utilisateur ne maîtrise PAS encore mais qui sont essentiels pour {target_job}.
Pour chacun, propose une ressource d'apprentissage concrète.

## Timeline réaliste
Plan sur 6-12 mois avec des jalons mensuels clairs. Adapte la charge en fonction de son niveau
d'études et son expérience. Un étudiant à temps plein peut avancer plus vite qu'un professionnel en poste.

Ton motivant et pragmatique. Sois PRÉCIS sur les noms de cours et certifications — pas de "un cours de ML" mais "Machine Learning Specialization par Andrew Ng sur Coursera"."""

    return call_gemini_api(prompt, cache_prefix="learning_path_v2")


# génère une bio professionnelle style LinkedIn à partir du profil
# on fait aussi un seul appel api, c'est suffisant pour 3-4 phrases
def generate_professional_bio(
    top_skills: Dict[str, float],
    recommended_job: str,
    user_inputs: Dict[str, str]
) -> str:
    if not top_skills:
        return "Profil en développement avec un fort potentiel d'évolution dans le domaine choisi."

    context = f"**Profil métier recommandé :** {recommended_job}\n\n"
    context += "**Points forts détectés :**\n"

    for bloc, score in sorted(top_skills.items(), key=lambda x: -x[1])[:3]:
        context += f"- {bloc} ({score:.0%})\n"

    projet_tech = user_inputs.get('projet_tech', '')
    if projet_tech:
        context += f"\n**Expérience clé :** {projet_tech[:200]}\n"

    prompt = f"""{context}

Rédige une bio professionnelle accrocheuse (style LinkedIn, section "À propos") en 3-4 phrases maximum.

Consignes :
- Mets en avant les compétences clés et le potentiel pour le métier {recommended_job}
- Ton enthousiaste mais professionnel
- Commence par une phrase d'accroche forte
- Termine par une ouverture sur les objectifs de carrière

Bio professionnelle (uniquement le texte, pas de titre ni préambule) :"""

    return call_gemini_api(prompt, cache_prefix="bio")


# tests rapides pour vérifier que tout marche
if __name__ == "__main__":
    print(f"Test du module GenAI Augmentation")
    print(f"Modele utilise : {MODEL_NAME}\n")

    # test enrichissement
    print("Test enrichissement texte court")
    short_text = "Python data"
    enriched = enrich_short_text(short_text)
    print(f"Original : {short_text}")
    print(f"Enrichi  : {enriched}\n")

    # test plan de progression
    print("Test generation plan de progression")
    weak_skills_test = {
        "Machine Learning": 0.35,
        "Deep Learning": 0.28,
        "NLP": 0.42
    }
    strong_skills_test_plan = {
        "Data Analysis": 0.75,
    }
    test_profile = {
        "likert_levels": {"data_analysis": "Avancé", "ml": "Intermédiaire", "dev": "Débutant"},
        "outils": {"Data & Analytics": ["Python", "Pandas", "SQL"]},
        "soft_skills": {"Rigueur": "Rigoureux et attentif aux détails"},
        "textes_libres": {"Projet dont il est fier": "Dashboard Power BI pour analyse des ventes"},
        "formation": {"Niveau d'etudes": "Bac+5 (Master)", "Experience": "0-2 ans"},
    }
    plan = generate_learning_path(
        weak_skills_test, strong_skills_test_plan, "Data Scientist", test_profile
    )
    print(plan)

    # test bio
    print("\nTest generation bio professionnelle")
    strong_skills_test = {
        "Data Analysis": 0.85,
        "Python": 0.78,
        "Visualisation": 0.72
    }
    bio = generate_professional_bio(
        strong_skills_test,
        "Data Analyst",
        {"projet_tech": "Création d'un dashboard Power BI pour analyse des ventes temps réel"}
    )
    print(bio)

    print("\nTous les tests termines.")
