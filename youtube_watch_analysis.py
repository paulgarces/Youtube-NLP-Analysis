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
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from nltk.corpus import stopwords
import nltk
from gensim.models import Word2Vec
import gensim.downloader as api

# Download NLTK stopwords if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Create folder for cluster outputs
os.makedirs("ClustersDataFrame", exist_ok=True)

# Load YouTube Watch History
path = "watch-history.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Define custom stopwords - REDUCED list keeping important signal words
custom_stopwords = set(stopwords.words('english')).union({
    "vs", "video", "official", "trailer", "new", "top", "best", 
    "moments", "ultimate", "ft", "watch", "watched", 
    "part", "shorts", "full", "complete",
    "original", "official", "hd", "4k", "short", "clip", "compilation"
})

# Keywords to KEEP even though they might be in common stopwords
words_to_keep = {
    "highlights", "music", "audio", "episode", "funny", 
    "gaming", "game", "live", "review", "tutorial", "how", 
    "show", "match", "tour", "house", "home", "apartment",
    "dance", "choreography", "song", "musical", "concert",
    "sports", "football", "soccer", "basketball", "nba", "nfl"
}

# Remove any words_to_keep from stopwords
custom_stopwords = custom_stopwords - words_to_keep

# Extract & Clean Titles
titles = []
original_titles = []
for entry in data:
    if entry.get("header") == "Youtube TV":  # Ignore YouTube TV
        continue
    title = entry.get("title", "").replace("Watched ", "").strip()
    if "details" in entry and any(d.get("name") == "From Google Ads" for d in entry["details"]):
        continue
    if title.startswith("https://") or title == "":
        continue
    original_titles.append(title)
    # Remove special characters and stopwords but keep important signal words
    title_cleaned = " ".join([word.lower() for word in re.sub(r"[^a-zA-Z0-9 ]", "", title).split() if word.lower() not in custom_stopwords])
    titles.append(title_cleaned)
    
print(f"Total valid videos: {len(titles)}")
print("Sample Titles:", original_titles[:5])
print("Sample Cleaned Titles:", titles[:5])

# Create hybrid embeddings (SBERT + TF-IDF)
print("\nGenerating embeddings...")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
sbert_embeddings = sbert_model.encode(titles, convert_to_numpy=True)

# Use TF-IDF with bigrams to capture phrases better
tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
tfidf_matrix = tfidf_vectorizer.fit_transform(titles).toarray()

# Combine SBERT + TF-IDF with weighted importance
X = np.hstack((sbert_embeddings * 0.8, tfidf_matrix * 0.2))  # Weight SBERT higher

# Normalize embeddings
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply Agglomerative Clustering - REDUCED to 8 clusters
num_clusters = 8  # Reduced from 12 to avoid fragmentation
agglo = AgglomerativeClustering(n_clusters=num_clusters)
clusters = agglo.fit_predict(X_scaled)

# Store results in a DataFrame
df = pd.DataFrame({"Original_Title": original_titles, "Cleaned_Title": titles, "Cluster": clusters})

# Count cluster sizes
cluster_sizes = df["Cluster"].value_counts().sort_index()
print("\nInitial Cluster Sizes:")
for cluster, size in cluster_sizes.items():
    print(f"Cluster {cluster}: {size} videos")

# Load pre-trained word vectors
print("Loading word vectors...")
word_vectors = api.load("glove-wiki-gigaword-100")

# Focused set of broad category names as targets
target_categories = [
    "gaming",
    "music", 
    "sports",
    "education",
    "food",
    "travel", 
    "movies & tv", 
    "news",
    "technology",
    "comedy", 
    "fitness",
    "entertainment",
    "housing & real estate",
    "podcasts"
]

# Function to extract meaningful keywords from a cluster
def get_meaningful_keywords(cluster_titles, n=30):
    # Get all words from the cluster
    all_words = " ".join(cluster_titles).split()
    word_counts = Counter(all_words)
    
    # Get the most common words that have at least 3 characters
    common_words = [word for word, count in word_counts.most_common(n*2) 
                   if len(word) >= 3 and word not in custom_stopwords and count > 1]
    
    # Select only words that appear in our word vectors
    vector_words = [word for word in common_words if word in word_vectors.key_to_index]
    
    # If not enough words with vectors, use the most common ones
    if len(vector_words) < n:
        return common_words[:n]
    return vector_words[:n]

# Function to generate cluster name from keywords
def generate_cluster_name(keywords, cluster_titles):
    if not keywords:
        return "Miscellaneous"
    
    # Get the 10 most significant keywords
    seed_words = keywords[:10]
    
    # Use the average of top keywords for better category matching
    keyword_embeddings = []
    for seed in seed_words:
        if seed in word_vectors.key_to_index:
            # Higher weight for very frequent terms
            count = " ".join(cluster_titles).count(seed)
            weight = min(count / 10, 3)  # Cap weight at 3x
            keyword_embeddings.extend([word_vectors[seed]] * int(weight))
    
    if not keyword_embeddings:
        return keywords[0].title()
        
    avg_embedding = np.mean(keyword_embeddings, axis=0)
    
    # Find the closest target category
    similarities = {}
    for category in target_categories:
        # Handle multi-word categories
        if " " in category:
            category_words = category.split()
            category_vector_words = [word for word in category_words if word in word_vectors.key_to_index]
            if not category_vector_words:
                continue
            category_embedding = np.mean([word_vectors[word] for word in category_vector_words], axis=0)
        elif category in word_vectors.key_to_index:
            category_embedding = word_vectors[category]
        else:
            continue
            
        # Calculate similarity using cosine similarity
        similarity = cosine_similarity(
            avg_embedding.reshape(1, -1), 
            category_embedding.reshape(1, -1)
        )[0][0]
        
        similarities[category] = similarity
    
    # Sort similarities and check if we have a clear winner
    sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    # DEBUG: Print the similarities for all categories
    print(f"\nSimilarities for keywords: {', '.join(seed_words[:5])}...")
    for cat, sim in sorted_sims[:5]:
        print(f"  {cat}: {sim:.4f}")
    
    if not sorted_sims:
        return keywords[0].title()
        
    best_category, best_sim = sorted_sims[0]
    
    # If the similarity is strong enough, use the category name
    if best_sim > 0.35:  # Slightly higher threshold
        return best_category.title()
    
    # Otherwise, use the most common keyword as fallback
    return keywords[0].title()

# Generate cluster names
cluster_names = {}
cluster_keywords = {}

print("\nAnalyzing clusters and generating names...")
for cluster_id in range(num_clusters):
    cluster_titles = df[df["Cluster"] == cluster_id]["Cleaned_Title"].tolist()
    keywords = get_meaningful_keywords(cluster_titles)
    cluster_keywords[cluster_id] = keywords
    
    # Generate a name for this cluster
    print(f"\nAnalyzing Cluster {cluster_id}...")
    cluster_name = generate_cluster_name(keywords, cluster_titles)
    cluster_names[cluster_id] = cluster_name

# Add cluster names to DataFrame
df["Cluster_Name"] = df["Cluster"].map(cluster_names)

# Print cluster analysis
print("\n=== Cluster Analysis ===")
for cluster_id in range(num_clusters):
    cluster_size = len(df[df["Cluster"] == cluster_id])
    print(f"\nCluster {cluster_id}: {cluster_names[cluster_id]} ({cluster_size} videos)")
    print(f"Top keywords: {', '.join(cluster_keywords[cluster_id][:10])}")
    print("Sample titles:")
    for title in df[df["Cluster"] == cluster_id]["Original_Title"].head(3).tolist():
        print(f"  • {title}")

# Post-process similar categories - merge clusters that got the same name
unique_names = set(cluster_names.values())
if len(unique_names) < num_clusters:
    print("\nDetected duplicate category names. Merging similar clusters...")
    
    # Create a mapping for new cluster IDs
    new_cluster_map = {}
    next_id = 0
    
    # First, group clusters by name
    name_to_clusters = {}
    for cluster_id, name in cluster_names.items():
        if name not in name_to_clusters:
            name_to_clusters[name] = []
        name_to_clusters[name].append(cluster_id)
    
    # Create mapping for each original cluster
    for name, cluster_ids in name_to_clusters.items():
        for cluster_id in cluster_ids:
            new_cluster_map[cluster_id] = next_id
        next_id += 1
    
    # Apply the mapping
    df["Merged_Cluster"] = df["Cluster"].map(new_cluster_map)
    df["Merged_Name"] = df["Cluster_Name"]  # Keep the same name
    
    # Update the number of clusters
    num_clusters = next_id
    
    # Print the merged clusters
    print("\n=== Merged Clusters ===")
    for cluster_id in range(num_clusters):
        cluster_df = df[df["Merged_Cluster"] == cluster_id]
        cluster_size = len(cluster_df)
        cluster_name = cluster_df["Merged_Name"].iloc[0]
        print(f"\nCluster {cluster_id}: {cluster_name} ({cluster_size} videos)")
        print("Sample titles:")
        for title in cluster_df["Original_Title"].head(3).tolist():
            print(f"  • {title}")
    
    # Use the merged clusters for saving
    df["Cluster"] = df["Merged_Cluster"]
    df["Cluster_Name"] = df["Merged_Name"]
    
    # Clean up temporary columns
    df = df.drop(columns=["Merged_Cluster", "Merged_Name"])

# Save each cluster to a separate CSV file
print("\nSaving results...")
for cluster_id in range(num_clusters):
    cluster_df = df[df["Cluster"] == cluster_id]
    safe_name = cluster_names.get(cluster_id, f"cluster_{cluster_id}").replace(" & ", "_").replace(" ", "_").lower()
    file_name = f"ClustersDataFrame/cluster_{cluster_id}_{safe_name}.csv"
    cluster_df.to_csv(file_name, index=False)
    print(f"Saved: {file_name}")

# Save overall results
df.to_csv("ClustersDataFrame/all_clusters.csv", index=False)
print("\n✅ Analysis complete! Results saved to the 'ClustersDataFrame' folder.")