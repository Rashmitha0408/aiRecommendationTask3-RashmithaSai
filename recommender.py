import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────
#  LOAD DATASET
# ─────────────────────────────────────────
df = pd.read_csv("raw_skills.csv")

# ─────────────────────────────────────────
#  WELCOME BANNER
# ─────────────────────────────────────────
print("=" * 55)
print("   🚀  TECH STACK RECOMMENDER ")
print("=" * 55)
print("\nThis tool maps your skills to the best career paths")
print("using TF-IDF Vectorization + Cosine Similarity.\n")

# ─────────────────────────────────────────
#  TAKE USER INPUT (minimum 3 skills)
# ─────────────────────────────────────────
print("Enter at least 3 skills you know.")
print("(Examples: Python, SQL, Machine_Learning, Docker, React)\n")

skills_input = []
for i in range(1, 4):
    skill = input(f"  Skill {i}: ").strip()
    skills_input.append(skill)

# Optional extra skills
print("\nWant to add more skills? (Press Enter to skip)")
while True:
    extra = input(f"  Extra skill (or press Enter to continue): ").strip()
    if extra == "":
        break
    skills_input.append(extra)

user_skills = " ".join(skills_input)

# ─────────────────────────────────────────
#  BUILD TF-IDF MATRIX
# ─────────────────────────────────────────
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["skills"])

# ─────────────────────────────────────────
#  VECTORIZE USER PROFILE
# ─────────────────────────────────────────
user_vector = vectorizer.transform([user_skills])

# ─────────────────────────────────────────
#  CALCULATE COSINE SIMILARITY
# ─────────────────────────────────────────
similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

# ─────────────────────────────────────────
#  SORT & FILTER — TOP 3
# ─────────────────────────────────────────
top_indices = similarity_scores.argsort()[::-1][:3]

# ─────────────────────────────────────────
#  DISPLAY RESULTS
# ─────────────────────────────────────────
print("\n" + "=" * 55)
print("   🎯  TOP 3 CAREER PATH RECOMMENDATIONS")
print("=" * 55)

if similarity_scores[top_indices[0]] == 0:
    print("\n⚠️  No matches found! Try using skills from the list.")
    print("   Examples: Python, SQL, Docker, React, AWS, Java\n")
else:
    for rank, idx in enumerate(top_indices, 1):
        role = df["job_role"][idx]
        score = similarity_scores[idx]
        match_percent = round(score * 100, 2)

        print(f"\n  #{rank}  {role}")
        print(f"       Match Score : {match_percent}%")
        print(f"       Skills Req  : {df['skills'][idx].replace('_', ' ')}")


