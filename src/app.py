# === Step 1: Import libraries ===
from sentence_transformers import SentenceTransformer, util
import numpy as np
# === Step 2: Define competency framework (blocks) ===
competency_blocks = {
    "Data Analysis": ["data cleaning", "data visualization", "statistics", "Python"],
    "Machine Learning": ["classification", "regression", "neural networks", "model evaluation"],
    "NLP": ["tokenization", "word embeddings", "transformers", "semantic analysis"]
}
# === Step 3: Collect user inputs (from questionnaire) ===
user_inputs = [
    "I have experience cleaning data with Python and building dashboards.",
    "I studied regression models and deep learning."
]
# === Step 4: Load SBERT model for embeddings ===
model = SentenceTransformer("all-MiniLM-L6-v2")
# Encode user inputs
user_embeddings = model.encode(user_inputs, convert_to_tensor=True)
# === Step 5: Calculate semantic similarity for each block ===
block_scores = {}
for block, competencies in competency_blocks.items():
    # Encode competency block phrases
    block_embeddings = model.encode(competencies, convert_to_tensor=True)
    # Compare each user input to competencies using cosine similarity
    similarities = util.cos_sim(user_embeddings, block_embeddings)
    # Take max similarity per user input and average across inputs
    max_similarities = [float(sim.max()) for sim in similarities]
    block_score = np.mean(max_similarities)
    block_scores[block] = block_score
# === Step 6: Weighted scoring formula (all equal weight here) ===

final_score = np.mean(list(block_scores.values()))
# === Step 7: Recommend job profile based on thresholds ===
if final_score >= 0.7:
    recommendation = "Data Scientist"
elif final_score >= 0.5:
    recommendation = "ML Engineer"
else:
    recommendation = "Entry-level Analyst"
# === Step 8: Display results ===
print("Block scores:", block_scores)
print("Final coverage score:", round(final_score, 2))
print("Recommended job profile:", recommendation)