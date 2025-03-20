import json
import pandas as pd
import re
import os
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter

# Create a folder to store cluster CSVs
os.makedirs("ClustersDataFrame", exist_ok=True)

# Load JSON file
path = "watch-history.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Define custom stopwords
custom_stopwords = {
    "best", "funny", "moments", "top", "new", "great", "ultimate", "official", "trailer",
    "the", "of", "and", "in", "to", "a", "is", "that", "it", "with", "for", "as", "on",
    "at", "by", "this", "but", "or", "an", "from", "not", "what", "all", "are", "was",
    "were", "when", "which", "there", "their", "they", "them", "these", "those", "its",
    "i", "you", "he", "she", "we", "my", "your", "his", "her", "our", "than", "how",
    "black", "to", "best", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "shorts",
    "us", "vs", "most", "full", "like", "me", "if", "so", "just", "get", "got", "premier",
    "nbc", "cup", "world"
}

# Define music-related keywords
music_keywords = {
    "music", "video", "ft", "feat", "featuring", "audio", "lyrics", "remix", 
    "song", "album", "official", "live", "dance", "choreography", "mv"
}

# Define sports-related keywords
sports_keywords = {
    "highlights", "league", "sports", "football", "soccer", "basketball", "nba", "nfl"
}

# Define apartment-related keywords
apartment_keywords = {
    "apartment", "tour", "nyc", "manhattan", "penthouse", "house", "home"
}

# Process titles with keyword boosting
original_titles = []
cleaned_titles = []
boosted_titles = []

for entry in data:
    if entry.get("header") == "Youtube TV":
        continue
    title = entry.get("title", "").replace("Watched ", "").strip()
    if "details" in entry and any(d.get("name") == "From Google Ads" for d in entry["details"]):
        continue
    if title.startswith("https://") or title == "":
        continue
    
    original_titles.append(title)
    
    # Basic cleaning
    title_cleaned = " ".join([word.lower() for word in re.sub(r"[^a-zA-Z0-9 ]", "", title).split()
                              if word.lower() not in custom_stopwords])
    cleaned_titles.append(title_cleaned)
    
    # Boosted title creation - duplicate important category keywords
    title_lower = title.lower()
    
    # Create boosted title by adding category markers
    boosted_title = title_cleaned
    
    # Check if title contains music-related terms
    music_count = sum(1 for keyword in music_keywords if keyword in title_lower)
    if music_count >= 2:  # If it has multiple music keywords
        boosted_title += " MUSICCATEGORY MUSICCATEGORY MUSICCATEGORY"
    
    # Check if title contains sports-related terms
    sports_count = sum(1 for keyword in sports_keywords if keyword in title_lower)
    if sports_count >= 2:
        boosted_title += " SPORTSCATEGORY SPORTSCATEGORY SPORTSCATEGORY"
    
    # Check if title contains apartment-related terms
    apartment_count = sum(1 for keyword in apartment_keywords if keyword in title_lower)
    if apartment_count >= 2:
        boosted_title += " APARTMENTCATEGORY APARTMENTCATEGORY APARTMENTCATEGORY"
    
    boosted_titles.append(boosted_title)

print(f"Total valid videos: {len(original_titles)}")

# Tokenize titles
titles_tokenized = [title.split() for title in boosted_titles]

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=titles_tokenized, vector_size=100, window=5, min_count=1, workers=4)

# Convert each title into an averaged word vector
title_vectors = []
for title in titles_tokenized:
    vectors = [word2vec_model.wv[word] for word in title if word in word2vec_model.wv]
    avg_vector = sum(vectors) / len(vectors) if vectors else [0] * 100
    title_vectors.append(avg_vector)

# Normalize data before clustering
scaler = StandardScaler()
X = scaler.fit_transform(title_vectors)

# Apply KMeans with 6 clusters
num_clusters = 6
kmeans = KMeans(n_clusters=num_clusters, random_state=42, init="k-means++", n_init=10)
clusters = kmeans.fit_predict(X)

# Store results in a DataFrame
dataframe = pd.DataFrame({
    "Original_Title": original_titles,
    "Cleaned_Title": cleaned_titles,
    "Cluster": clusters
})

# Auto-label clusters using most common words
cluster_labels = {}
for cluster_num in range(num_clusters):
    cluster_titles = dataframe[dataframe["Cluster"] == cluster_num]["Cleaned_Title"].tolist()
    words = [word for title in cluster_titles for word in title.split()]
    most_common_words = Counter(words).most_common(5)
    cluster_name = " / ".join([word for word, count in most_common_words])
    cluster_labels[cluster_num] = cluster_name

# Add labeled clusters to DataFrame
dataframe["Cluster_Name"] = dataframe["Cluster"].map(cluster_labels)

# Print cluster categories
print("\n### Cluster Names ###\n")
for cluster, label in cluster_labels.items():
    count = len(dataframe[dataframe["Cluster"] == cluster])
    print(f"Cluster {cluster} → {label} ({count} videos)")
    # Print sample titles
    for title in dataframe[dataframe["Cluster"] == cluster]["Original_Title"].head(3).tolist():
        print(f"  • {title}")
    print()

# Save each cluster separately
for cluster_num in range(num_clusters):
    file_name = f"ClustersDataFrame/cluster_{cluster_num}.csv"
    df = dataframe[dataframe["Cluster"] == cluster_num]
    df.to_csv(file_name, index=False)

print("\n✅ Clusters saved in 'ClustersDataFrame' folder")