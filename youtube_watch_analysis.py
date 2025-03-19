# import json
# import pandas as pd
# import re
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
# import os

# os.makedirs("ClustersDataFrame", exist_ok=True)

# path = "watch-history.json"
# with open(path, "r", encoding="utf-8") as f:
#     data = json.load(f)

# titles = []
# for entry in data:
#     if entry.get("header") == "Youtube TV":
#         continue

#     title = entry.get("title", "").replace("Watched ", "").strip()

#     if "details" in entry and any(d.get("name") == "From Google Ads" for d in entry["details"]):
#         continue

#     if title.startswith("https://") or title == "":
#         continue

#     titles.append(title)

# print(f"Total valid videos: {len(titles)}")
# print("Sample Titles:", titles[:10])

# titles_cleaned = [re.sub(r"[^a-zA-Z0-9 ]", "", title.lower()) for title in titles]
# vectorizer = TfidfVectorizer(stop_words="english")
# X = vectorizer.fit_transform(titles_cleaned)

# num_clusters = 5
# kmeans = KMeans(n_clusters=num_clusters, random_state=42, init="k-means++", n_init=10)
# clusters = kmeans.fit_predict(X)

# dataframe = pd.DataFrame({"Title": titles, "Cluster": clusters})

# cluster_dataframes = {i: dataframe[dataframe["Cluster"] == i] for i in range(num_clusters)}

# for cluster_num in range(num_clusters):
#     file_name = f"ClustersDataFrame/cluster_{cluster_num}.csv"
#     df = dataframe[dataframe["Cluster"] == cluster_num]
#     df.to_csv(file_name, index=False)

import json
import pandas as pd
import re
import os
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Create folder for cluster outputs
os.makedirs("ClustersDataFrame", exist_ok=True)

# Load YouTube Watch History
path = "watch-history.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Define stopwords to remove irrelevant words
custom_stopwords = {
    "the", "to", "a", "in", "for", "how", "you", "and", "vs", "video", "official",
    "trailer", "new", "top", "best", "funny", "moments", "ultimate", "highlights",
    "ft", "music", "audio", "watch", "watched", "episode", "clips", "part", "shorts"
}

# Extract & Clean Titles
titles = []
for entry in data:
    if entry.get("header") == "Youtube TV":  # Ignore YouTube TV
        continue
    title = entry.get("title", "").replace("Watched ", "").strip()
    if "details" in entry and any(d.get("name") == "From Google Ads" for d in entry["details"]):
        continue
    if title.startswith("https://") or title == "":
        continue
    # Remove special characters and stopwords
    title_cleaned = " ".join([word.lower() for word in re.sub(r"[^a-zA-Z0-9 ]", "", title).split() if word not in custom_stopwords])
    titles.append(title)  # Keep original title for readability
    
cleaned_titles = [" ".join([word.lower() for word in re.sub(r"[^a-zA-Z0-9 ]", "", title).split() if word not in custom_stopwords]) for title in titles]

print(f"Total valid videos: {len(titles)}")
print("Sample Titles:", titles[:5])

# Define predetermined categories with keywords
predefined_categories = {
    "Music": ["song", "music", "dance", "choreography", "album", "concert", "lyrics", "remix", "beat", "instrumental", "singing", "vocalist", "band", "rap", "hip hop", "jazz", "rock", "pop", "edm", "dj"],
    "Gaming": ["game", "gaming", "playthrough", "gameplay", "multiplayer", "speedrun", "let's play", "minecraft", "fortnite", "valorant", "league of legends", "dota", "strategy", "rpg", "fps", "walkthrough", "console", "pc game"],
    "Sports": ["football", "basketball", "soccer", "nba", "nfl", "mlb", "fifa", "match", "highlights", "sport", "athletic", "champion", "tournament", "boxing", "cricket", "tennis", "golf", "olympics", "world cup"],
    "Movies & TV": ["movie", "film", "series", "episode", "season", "tv show", "cinema", "documentary", "netflix", "hulu", "disney", "actor", "director", "plot", "review", "trailer", "scene", "character", "production"],
    "Comedy": ["comedy", "funny", "joke", "stand up", "humor", "prank", "parody", "sketch", "comedian", "laugh", "meme", "roast", "sitcom"],
    "Education": ["tutorial", "learn", "education", "course", "lecture", "professor", "school", "university", "college", "teach", "training", "exam", "lesson", "study", "explanation", "guide", "how to"],
    "Technology": ["tech", "review", "unboxing", "gadget", "smartphone", "computer", "laptop", "software", "hardware", "programming", "coding", "development", "ai", "machine learning", "data science", "algorithm"],
    "Travel & Lifestyle": ["travel", "vlog", "lifestyle", "tour", "apartment", "house", "home", "mansion", "penthouse", "renovation", "decoration", "design", "trip", "journey", "vacation", "hotel", "resort", "city", "country"],
    "Food & Cooking": ["food", "recipe", "cooking", "chef", "kitchen", "baking", "restaurant", "cuisine", "meal", "dish", "ingredient", "taste", "flavor", "culinary", "diet", "nutrition"],
    "News & Politics": ["news", "politics", "analysis", "election", "debate", "president", "government", "campaign", "senator", "congress", "parliament", "candidate", "vote", "political", "policy", "issue"]
}

# Function to classify a title
def classify_title(title, categories):
    title_lower = title.lower()
    
    # Check for direct keyword matches
    scores = {}
    for category, keywords in categories.items():
        score = sum(1 for keyword in keywords if keyword.lower() in title_lower)
        scores[category] = score
    
    # If we have a clear match, return it
    max_score = max(scores.values())
    if max_score > 0:
        best_categories = [cat for cat, score in scores.items() if score == max_score]
        return best_categories[0]
    
    # If no direct match, return "Other"
    return "Other"

# Use SBERT for semantic understanding of titles
print("Loading SBERT model and generating embeddings...")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
title_embeddings = sbert_model.encode(cleaned_titles, convert_to_numpy=True)

# Generate category embeddings (average of multiple keywords)
category_embeddings = {}
for category, keywords in predefined_categories.items():
    keyword_embeddings = sbert_model.encode(keywords, convert_to_numpy=True)
    category_embeddings[category] = np.mean(keyword_embeddings, axis=0)

# Classify using semantic similarity as fallback for keyword matching
classifications = []
for i, title in enumerate(titles):
    # First try keyword matching
    category = classify_title(title, predefined_categories)
    
    # If no match, use semantic similarity
    if category == "Other":
        title_embedding = title_embeddings[i].reshape(1, -1)
        similarities = {}
        for cat, cat_embedding in category_embeddings.items():
            similarity = cosine_similarity(title_embedding, cat_embedding.reshape(1, -1))[0][0]
            similarities[cat] = similarity
        
        category = max(similarities.items(), key=lambda x: x[1])[0]
    
    classifications.append(category)

# Store results in a DataFrame
dataframe = pd.DataFrame({"Title": titles, "Category": classifications})

# Print category distribution
print("\n### Category Distribution ###\n")
print(dataframe["Category"].value_counts())

# Save each category separately into CSV files
for category in set(classifications):
    file_name = f"ClustersDataFrame/category_{category.replace(' & ', '_').replace(' ', '_').lower()}.csv"
    df = dataframe[dataframe["Category"] == category]
    df.to_csv(file_name, index=False)

print("\n✅ Categories saved in 'ClustersDataFrame' folder")

# Show examples from each category
print("\n### Examples from each category ###\n")
for category in set(classifications):
    print(f"--- {category} ---")
    examples = dataframe[dataframe["Category"] == category]["Title"].sample(min(5, len(dataframe[dataframe["Category"] == category]))).tolist()
    for ex in examples:
        print(f"  • {ex}")
    print()