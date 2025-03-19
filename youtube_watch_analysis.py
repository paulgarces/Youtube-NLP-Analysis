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
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic

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
    
    titles.append(title_cleaned)

print(f"Total valid videos: {len(titles)}")
print("Sample Titles:", titles[:10])

# **Generate Hybrid Embeddings (TF-IDF + SBERT)**
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")  # Small but powerful model
sbert_embeddings = sbert_model.encode(titles, convert_to_numpy=True)  # Get SBERT embeddings

# Use TF-IDF to emphasize important words
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(titles).toarray()

# Combine SBERT + TF-IDF
X = np.hstack((sbert_embeddings, tfidf_matrix))

# Normalize embeddings
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply Agglomerative Clustering (instead of K-Means)
num_clusters = 10  # **Increased to separate topics better**
agglo = AgglomerativeClustering(n_clusters=num_clusters)
clusters = agglo.fit_predict(X_scaled)

# Store results in a DataFrame
dataframe = pd.DataFrame({"Title": titles, "Cluster": clusters})

# ✅ **Use BERTopic with Pre-Defined Categories**
topic_model = BERTopic(nr_topics="auto")  # Let it auto-detect categories
topics, _ = topic_model.fit_transform(titles)

# Map clusters to BERTopic topics
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

# Assign auto-detected cluster names
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