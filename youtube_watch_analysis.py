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
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from bertopic import BERTopic

# Create a folder for cluster CSV files
os.makedirs("ClustersDataFrame", exist_ok=True)

# Load YouTube Watch History
path = "watch-history.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Custom stopwords to filter out common words
custom_stopwords = {
    "the", "to", "a", "in", "for", "how", "you", "and", "vs", "video", "official",
    "trailer", "new", "top", "best", "funny", "moments", "ultimate", "highlights",
    "ft", "music", "audio", "watch", "watched"
}

# Extract cleaned video titles
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
    
    titles.append(title_cleaned)

print(f"Total valid videos: {len(titles)}")
print("Sample Titles:", titles[:10])

# Tokenize titles into lists of words
titles_tokenized = [title.split() for title in titles]

# Train Word2Vec model on tokenized words
word2vec_model = Word2Vec(sentences=titles_tokenized, vector_size=100, window=5, min_count=1, workers=4)

# Convert each title into an averaged word vector
title_vectors = []
for title in titles_tokenized:
    vectors = [word2vec_model.wv[word] for word in title if word in word2vec_model.wv]
    avg_vector = np.mean(vectors, axis=0) if vectors else np.zeros(100)  # Handle empty cases
    title_vectors.append(avg_vector)

# Normalize data before clustering
scaler = StandardScaler()
X = scaler.fit_transform(title_vectors)

# Apply KMeans Clustering
num_clusters = 8
kmeans = KMeans(n_clusters=num_clusters, random_state=42, init="k-means++", n_init=10)
clusters = kmeans.fit_predict(X)

# Store results in a DataFrame
dataframe = pd.DataFrame({"Title": titles, "Cluster": clusters})

# ✅ **NEW: Auto-Generate Cluster Labels with BERTopic**
topic_model = BERTopic()
topics, _ = topic_model.fit_transform(titles)  # Train BERTopic on raw titles

# Map KMeans clusters to BERTopic-generated topics
cluster_names = {}
for cluster in range(num_clusters):
    cluster_titles = dataframe[dataframe["Cluster"] == cluster]["Title"].tolist()
    
    if cluster_titles:
        topic_idx, _ = topic_model.find_topics(" ".join(cluster_titles), top_n=1)
        
        if topic_idx:
            top_words = topic_model.get_topic(topic_idx[0])  # Get top words for this topic
            cluster_names[cluster] = " / ".join([word for word, _ in top_words[:3]])  # Use top 3 words
        else:
            cluster_names[cluster] = "Unknown"

# Assign meaningful cluster names
dataframe["Cluster Name"] = dataframe["Cluster"].map(cluster_names)

# Print the final auto-labeled cluster names
print("\n### Auto-Generated Cluster Categories ###\n")
for cluster, label in cluster_names.items():
    print(f"Cluster {cluster} → {label}")

# Save each cluster separately into CSV files
for cluster_num in range(num_clusters):
    file_name = f"ClustersDataFrame/cluster_{cluster_num}.csv"
    df = dataframe[dataframe["Cluster"] == cluster_num]
    df.to_csv(file_name, index=False)

print("\n✅ Clusters saved in 'ClustersDataFrame' folder")