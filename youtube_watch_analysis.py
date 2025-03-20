import json
import pandas as pd
import re
import os
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

# Define custom stopwords (remove generic words like "best", "funny", "moments")
custom_stopwords = {"best", "funny", "moments", "top", "new", "great", "ultimate", "official", "trailer"}

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

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=titles_tokenized, vector_size=100, window=5, min_count=1, workers=4)

# Convert each title into an averaged word vector
title_vectors = []
for title in titles_tokenized:
    vectors = [word2vec_model.wv[word] for word in title if word in word2vec_model.wv]
    avg_vector = sum(vectors) / len(vectors) if vectors else [0] * 100  # Handle empty cases
    title_vectors.append(avg_vector)

# Normalize data before clustering
scaler = StandardScaler()
X = scaler.fit_transform(title_vectors)

# Apply KMeans with more clusters
num_clusters = 7  # Increase clusters for better separation
kmeans = KMeans(n_clusters=num_clusters, random_state=42, init="k-means++", n_init=10)
clusters = kmeans.fit_predict(X)

# Store results in a DataFrame
dataframe = pd.DataFrame({"Title": titles, "Cluster": clusters})

# Auto-label clusters using most common words
cluster_labels = {}

for cluster_num in range(num_clusters):
    cluster_titles = dataframe[dataframe["Cluster"] == cluster_num]["Title"].tolist()
    
    words = [word for title in cluster_titles for word in title.split()]
    most_common_words = Counter(words).most_common(5)  # Get top words in cluster
    
    cluster_name = " / ".join([word for word, _ in most_common_words])  # Assign category name
    cluster_labels[cluster_num] = cluster_name

# Add labeled clusters to DataFrame
dataframe["Cluster Name"] = dataframe["Cluster"].map(cluster_labels)

# Print cluster categories
print("\n### Cluster Names ###\n")
for cluster, label in cluster_labels.items():
    print(f"Cluster {cluster} → {label}")

# Save each cluster separately
for cluster_num in range(num_clusters):
    file_name = f"ClustersDataFrame/cluster_{cluster_num}.csv"
    df = dataframe[dataframe["Cluster"] == cluster_num]
    df.to_csv(file_name, index=False)

print("\n✅ Clusters saved in 'ClustersDataFrame' folder")


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