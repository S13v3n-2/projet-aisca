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
    "max_output_tokens": 8192,
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
def call_gemini_api(prompt: str, cache_prefix: str, bypass_cache: bool = False) -> str:
    if not GEMINI_API_KEY:
        return "Erreur API Gemini : cle API manquante, cree un .env avec GEMINI_API_KEY=ta_cle"

    cache_key = get_cache_key(prompt, prefix=cache_prefix)

    if not bypass_cache:
        cached = load_from_cache(cache_key)
        if cached:
            # garde-fou : si le cache contient une réponse tronquée, on l'ignore
            stripped = cached.strip()
            is_truncated = (
                len(stripped) > 100 and
                stripped[-1] not in '.!?*_)»"' and
                not stripped.endswith('---') and
                not stripped.endswith('```')
            )
            if not is_truncated:
                print(f"Cache hit pour {cache_prefix}")
                return cached
            else:
                print(f"Cache tronqué détecté pour {cache_prefix}, on régénère")

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


# génère un plan de progression personnalisé pour les blocs faibles
# nouvelle signature : accepte weak_skills, strong_skills, target_job, user_profile_data (dict)
# on fait un seul appel api pour tout le plan, pas un par compétence
def generate_learning_path(
    weak_skills: Dict[str, float],
    strong_skills: Dict[str, float],
    target_job: str,
    user_profile_data: Dict
) -> str:
    if not weak_skills:
        return "Vous maîtrisez déjà tous les blocs requis pour ce métier."

    # construction du contexte enrichi depuis user_profile_data
    likert   = user_profile_data.get("likert_levels", {})
    outils   = user_profile_data.get("outils", {})
    textes   = user_profile_data.get("textes_libres", {})
    form     = user_profile_data.get("formation", {})
    comps    = user_profile_data.get("comps_a_dev", [])

    outils_flat = [o for lst in outils.values() for o in lst if lst]
    projet      = textes.get("Projet dont il est fier", "")
    objectif    = textes.get("Objectif de carriere", "")

    context  = f"Métier cible : {target_job}\n"
    context += "Formation : " + form.get("Niveau d'etudes", "") + " en " + form.get("Domaine de formation", "") + "\n"
    context += f"Expérience : {form.get('Experience professionnelle', '')}\n"
    if outils_flat:
        context += f"Outils maîtrisés : {', '.join(outils_flat[:10])}\n"
    if projet:
        context += f"Projet réalisé : {projet[:200]}\n"
    if objectif:
        context += f"Objectif carrière : {objectif[:150]}\n"

    context += "\nBlocs maîtrisés :\n"
    for bloc, score in sorted(strong_skills.items(), key=lambda x: -x[1]):
        context += f"- {bloc} : {score:.0%}\n"

    context += "\nBlocs à développer (par priorité) :\n"
    for bloc, score in sorted(weak_skills.items(), key=lambda x: x[1]):
        context += f"- {bloc} : {score:.0%}\n"

    if comps:
        context += f"\nCompétences spécifiques les plus faibles : {', '.join(comps[:6])}\n"

    prompt = f"""{context}

En tant qu'expert en orientation professionnelle, génère un plan de progression personnalisé.

RÈGLES DE FORMAT — OBLIGATOIRES :
- Utilise UNIQUEMENT du Markdown pur. JAMAIS de HTML (pas de <div>, <p>, <span>, ni aucune balise).
- JAMAIS de backticks (`code`) — ni pour les noms d'outils, ni pour les compétences, ni pour quoi que ce soit.
- Structure avec des titres ## pour chaque section principale.
- Utilise des listes - pour les actions concrètes.
- Utilise **gras** uniquement pour les titres de sous-sections (ex: **Pourquoi c'est prioritaire**).
- Chaque phrase doit être complète et se terminer par un point.

STRUCTURE DU PLAN :

## Diagnostic personnalisé
En 2-3 phrases, analyse le profil et identifie les axes prioritaires.

## Plan de progression détaillé
Pour chaque bloc à développer, une sous-section ## avec :
- Pourquoi c'est prioritaire
- Ce que l'utilisateur a déjà
- Ce qu'il doit apprendre (cours précis + plateforme)
- Projet pratique concret
- Certification recommandée si pertinente

## Outils à acquérir
Liste les outils complémentaires essentiels non encore maîtrisés.

Ton : motivant, pragmatique, personnalisé. Maximum 800 mots. Termine OBLIGATOIREMENT par une phrase complète."""

    bypass = user_profile_data.get("force_regen", False)
    return call_gemini_api(prompt, cache_prefix="learning_path_v2", bypass_cache=bypass)


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
    plan = generate_learning_path(
        weak_skills_test,
        "Data Scientist",
        "Profil junior avec bases en Python et statistiques, expérience en analyse de données"
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