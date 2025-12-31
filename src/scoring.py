# === Step 1: Import libraries ===
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import json

with open("../data/referentiel.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. Création de l'index des compétences
# On itère sur data['blocs'] qui est la liste des blocs
competences_index = {
    comp['id']: comp
    for bloc in data['blocs']
    for comp in bloc['competences']
}

# Test d'accès
print(f"Nombre de compétences indexées : {len(competences_index)}")

