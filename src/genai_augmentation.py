"""
Module d'augmentation par GenAI (RAG) - Respect Free Tier.
Intègre Google Gemini 2.5 Flash avec caching automatique.
"""

import os
import json
import hashlib
from typing import Optional, Dict
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

# ============================================================================
# CHARGEMENT VARIABLES D'ENVIRONNEMENT
# ============================================================================

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(
        "❌ GEMINI_API_KEY introuvable dans .env\n"
        "Crée un fichier .env avec : GEMINI_API_KEY=ta_clé_ici"
    )

# ============================================================================
# CONFIGURATION
# ============================================================================

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

genai.configure(api_key=GEMINI_API_KEY)

# ✅ MODÈLE CORRECT : gemini-2.5-flash (stable, juin 2025, 1M tokens)
MODEL_NAME = "gemini-2.5-flash"

generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 2048,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

MAX_SHORT_TEXT_LENGTH = 15
MIN_ENRICHMENT_LENGTH = 5


# ============================================================================
# CACHING SYSTÈME
# ============================================================================

def get_cache_key(prompt: str, prefix: str = "") -> str:
    """Génère une clé de cache unique basée sur le prompt."""
    content = f"{prefix}:{prompt}"
    return hashlib.sha256(content.encode()).hexdigest()


def load_from_cache(cache_key: str) -> Optional[str]:
    """Charge une réponse depuis le cache si elle existe."""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("response")
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ Erreur lecture cache : {e}")
            return None
    return None


def save_to_cache(cache_key: str, response: str):
    """Sauvegarde une réponse dans le cache."""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({"response": response}, f, ensure_ascii=False, indent=2)
    except IOError as e:
        print(f"⚠️ Erreur écriture cache : {e}")


# ============================================================================
# HELPER : Appel API Gemini avec gestion d'erreurs
# ============================================================================

def call_gemini_api(prompt: str, cache_prefix: str) -> str:
    """
    Fonction générique pour appeler l'API Gemini avec cache et gestion d'erreurs.
    """
    cache_key = get_cache_key(prompt, prefix=cache_prefix)
    cached = load_from_cache(cache_key)

    if cached:
        print(f"✅ Cache hit pour {cache_prefix}")
        return cached

    try:
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        print(f"🔄 Appel API Gemini ({cache_prefix})...")
        response = model.generate_content(prompt)

        if not response or not response.text:
            if hasattr(response, 'prompt_feedback'):
                return f"⚠️ Réponse bloquée : {response.prompt_feedback}"
            return "⚠️ Pas de réponse générée."

        generated_text = response.text.strip()
        save_to_cache(cache_key, generated_text)
        print(f"✅ Réponse générée et mise en cache")

        return generated_text

    except Exception as e:
        error_msg = f"⚠️ Erreur API Gemini : {type(e).__name__} - {str(e)}"
        print(error_msg)
        return error_msg


# ============================================================================
# AUGMENTATION TEXTE (EF4.1)
# ============================================================================

def enrich_short_text(user_text: str) -> str:
    """
    Enrichit un texte utilisateur trop court via GenAI (usage conditionnel).

    Args:
        user_text: Texte d'origine

    Returns:
        Texte enrichi ou texte d'origine si déjà suffisant
    """
    if not user_text or not user_text.strip():
        return user_text

    word_count = len(user_text.split())

    if word_count >= MIN_ENRICHMENT_LENGTH:
        print(f"ℹ️ Texte suffisamment long ({word_count} mots), pas d'enrichissement")
        return user_text

    print(f"🔍 Enrichissement texte court ({word_count} mots)...")

    prompt = f"""Tu es un expert en orientation professionnelle. 
L'utilisateur a donné une réponse très courte : "{user_text}"

Réécris cette phrase en ajoutant du contexte technique professionnel pertinent, 
sans inventer de compétences non mentionnées. Maximum 3 phrases.

Réponse enrichie (uniquement le texte, pas de préambule) :"""

    enriched = call_gemini_api(prompt, cache_prefix="enrich")

    if enriched.startswith("⚠️"):
        print(f"⚠️ Échec enrichissement, texte original conservé")
        return user_text

    return enriched


# ============================================================================
# PLAN DE PROGRESSION (EF4.2)
# ============================================================================

def generate_learning_path(
    weak_skills: Dict[str, float],
    target_job: str,
    user_profile: str
) -> str:
    """
    Génère un plan de progression personnalisé (1 seul appel API).

    Args:
        weak_skills: {bloc_name: score} des compétences faibles
        target_job: Métier cible
        user_profile: Résumé du profil utilisateur

    Returns:
        Plan de progression formaté en Markdown
    """
    if not weak_skills:
        return "✨ **Félicitations !** Vous maîtrisez déjà tous les blocs requis pour ce métier."

    context = f"**Métier cible :** {target_job}\n\n"
    context += f"**Profil utilisateur :** {user_profile[:500]}...\n\n"
    context += "**Compétences à développer prioritairement :**\n"

    for bloc, score in sorted(weak_skills.items(), key=lambda x: x[1])[:5]:
        context += f"- {bloc} (score actuel : {score:.0%})\n"

    prompt = f"""{context}

En tant qu'expert en orientation Data/IA/Business, génère un plan de progression structuré :

## 1. Compétences prioritaires
Liste les 3 compétences à développer en PREMIER pour le métier {target_job}.

## 2. Ressources concrètes
Pour chaque compétence, propose :
- Un cours en ligne (nom précis + plateforme : Coursera, Udemy, OpenClassrooms, etc.)
- Un projet pratique à réaliser
- Une certification pertinente si applicable

## 3. Timeline réaliste
Étapes sur 6-12 mois avec jalons mensuels clairs.

Utilise un ton motivant et pragmatique. Format Markdown strict."""

    return call_gemini_api(prompt, cache_prefix="learning_path")


# ============================================================================
# BIO PROFESSIONNELLE (EF4.3)
# ============================================================================

def generate_professional_bio(
    top_skills: Dict[str, float],
    recommended_job: str,
    user_inputs: Dict[str, str]
) -> str:
    """
    Génère une bio professionnelle type Executive Summary (1 seul appel).

    Args:
        top_skills: Compétences fortes détectées
        recommended_job: Métier recommandé #1
        user_inputs: Champs du questionnaire

    Returns:
        Bio professionnelle concise (3-4 phrases)
    """
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


# ============================================================================
# TEST UNITAIRE
# ============================================================================

if __name__ == "__main__":
    print(f"🧪 Test du module GenAI Augmentation")
    print(f"📌 Modèle utilisé : {MODEL_NAME}\n")
    print("="*70 + "\n")

    # Test 1 : Enrichissement
    print("1️⃣ TEST ENRICHISSEMENT TEXTE COURT")
    print("-" * 70)
    short_text = "Python data"
    enriched = enrich_short_text(short_text)
    print(f"📝 Original : {short_text}")
    print(f"✨ Enrichi  : {enriched}")
    print("\n" + "="*70 + "\n")

    # Test 2 : Plan de progression
    print("2️⃣ TEST GÉNÉRATION PLAN DE PROGRESSION")
    print("-" * 70)
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
    print("\n" + "="*70 + "\n")

    # Test 3 : Bio professionnelle
    print("3️⃣ TEST GÉNÉRATION BIO PROFESSIONNELLE")
    print("-" * 70)
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
    print("\n" + "="*70 + "\n")

    print("✅ Tous les tests terminés avec succès !")
