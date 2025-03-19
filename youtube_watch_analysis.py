import json
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import os

os.makedirs("ClustersDataFrame", exist_ok=True)

path = "watch-history.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

titles = []
for entry in data:
    if entry.get("header") == "Youtube TV":
        continue

    title = entry.get("title", "").replace("Watched ", "").strip()

    if "details" in entry and any(d.get("name") == "From Google Ads" for d in entry["details"]):
        continue

    if title.startswith("https://") or title == "":
        continue

    titles.append(title)

print(f"Total valid videos: {len(titles)}")
print("Sample Titles:", titles[:10])

titles_cleaned = [re.sub(r"[^a-zA-Z0-9 ]", "", title.lower()) for title in titles]
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(titles_cleaned)

num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42, init="k-means++", n_init=10)
clusters = kmeans.fit_predict(X)

dataframe = pd.DataFrame({"Title": titles, "Cluster": clusters})

cluster_dataframes = {i: dataframe[dataframe["Cluster"] == i] for i in range(num_clusters)}

for cluster_num in range(num_clusters):
    file_name = f"ClustersDataFrame/cluster_{cluster_num}.csv"
    df = dataframe[dataframe["Cluster"] == cluster_num]
    df.to_csv(file_name, index=False)