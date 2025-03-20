# Paul Garces - YouTube Watch History Analysis
# This .py script reads the watch history from a JSON file and performs clustering on the video titles

import json
import pandas as pd
import re
import os
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter

os.makedirs("ClustersDataFrame", exist_ok=True)

path = "watch-history.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

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

music_keywords = {
    "music", "video", "ft", "feat", "featuring", "audio", "lyrics", "remix", 
    "song", "album", "official", "live", "dance", "choreography", "mv"
}

sports_keywords = {
    "highlights", "league", "sports", "football", "soccer", "basketball", "nba", "nfl"
}

apartment_keywords = {
    "apartment", "tour", "nyc", "manhattan", "penthouse", "house", "home"
}

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
    
    title_cleaned = " ".join([word.lower() for word in re.sub(r"[^a-zA-Z0-9 ]", "", title).split()
                              if word.lower() not in custom_stopwords])
    cleaned_titles.append(title_cleaned)
    
    title_lower = title.lower()
    
    boosted_title = title_cleaned
    
    music_count = sum(1 for keyword in music_keywords if keyword in title_lower)
    if music_count >= 2:
        boosted_title += " MUSICCATEGORY MUSICCATEGORY MUSICCATEGORY"
    
    sports_count = sum(1 for keyword in sports_keywords if keyword in title_lower)
    if sports_count >= 2:
        boosted_title += " SPORTSCATEGORY SPORTSCATEGORY SPORTSCATEGORY"
    
    apartment_count = sum(1 for keyword in apartment_keywords if keyword in title_lower)
    if apartment_count >= 2:
        boosted_title += " APARTMENTCATEGORY APARTMENTCATEGORY APARTMENTCATEGORY"
    
    boosted_titles.append(boosted_title)

print(f"Total valid videos: {len(original_titles)}")

titles_tokenized = [title.split() for title in boosted_titles]

word2vec_model = Word2Vec(sentences=titles_tokenized, vector_size=100, window=5, min_count=1, workers=4)

title_vectors = []
for title in titles_tokenized:
    vectors = [word2vec_model.wv[word] for word in title if word in word2vec_model.wv]
    avg_vector = sum(vectors) / len(vectors) if vectors else [0] * 100
    title_vectors.append(avg_vector)

scaler = StandardScaler()
X = scaler.fit_transform(title_vectors)

num_clusters = 6
kmeans = KMeans(n_clusters=num_clusters, random_state=42, init="k-means++", n_init=10)
clusters = kmeans.fit_predict(X)

dataframe = pd.DataFrame({
    "Original_Title": original_titles,
    "Cleaned_Title": cleaned_titles,
    "Cluster": clusters
})

cluster_labels = {}
for cluster_num in range(num_clusters):
    cluster_titles = dataframe[dataframe["Cluster"] == cluster_num]["Cleaned_Title"].tolist()
    words = [word for title in cluster_titles for word in title.split()]
    most_common_words = Counter(words).most_common(5)
    cluster_name = " / ".join([word for word, count in most_common_words])
    cluster_labels[cluster_num] = cluster_name

dataframe["Cluster_Name"] = dataframe["Cluster"].map(cluster_labels)

print("\n### Cluster Names ###\n")
for cluster, label in cluster_labels.items():
    count = len(dataframe[dataframe["Cluster"] == cluster])
    print(f"Cluster {cluster} → {label} ({count} videos)")
    for title in dataframe[dataframe["Cluster"] == cluster]["Original_Title"].head(3).tolist():
        print(f"  • {title}")
    print()

for cluster_num in range(num_clusters):
    file_name = f"ClustersDataFrame/cluster_{cluster_num}.csv"
    df = dataframe[dataframe["Cluster"] == cluster_num]
    
    sample_size = min(2000, len(df))
    df_to_save = df.head(sample_size)
    
    df_to_save.to_csv(file_name, index=False)
    print(f"Saved {sample_size} rows for Cluster {cluster_num}")

print("\n Clusters saved in 'ClustersDataFrame' file!")