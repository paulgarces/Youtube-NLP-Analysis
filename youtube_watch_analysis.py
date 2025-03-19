import json
import pandas as pd
import re
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

# Create folder for cluster outputs
os.makedirs("ClustersDataFrame", exist_ok=True)

# Load YouTube Watch History
path = "watch-history.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Define minimal stopwords - only the most basic ones
# We want to keep most content words to help with clustering
minimal_stopwords = {
    "the", "of", "and", "in", "to", "a", "is", "that", "it", "with", "for", "as", "on",
    "at", "by", "this", "but", "or", "an", "from", "not", "what", "all", "are", "was", 
    "were", "when", "which", "there", "their", "they", "them", "these", "those", "its",
    "i", "you", "he", "she", "we", "my", "your", "his", "her", "our", "than"
}

# Process and clean titles
original_titles = []
cleaned_titles = []

for entry in data:
    if entry.get("header") == "Youtube TV":
        continue
    title = entry.get("title", "").replace("Watched ", "").strip()
    if "details" in entry and any(d.get("name") == "From Google Ads" for d in entry["details"]):
        continue
    if title.startswith("https://") or title == "":
        continue
    
    original_titles.append(title)
    
    # Minimal cleaning - just remove special characters and basic stopwords
    title_cleaned = " ".join([word.lower() for word in re.sub(r"[^a-zA-Z0-9 ]", "", title).split() 
                              if word.lower() not in minimal_stopwords])
    cleaned_titles.append(title_cleaned)

print(f"Total valid videos: {len(original_titles)}")
print("Sample Titles:", original_titles[:3])

# Create TF-IDF features with bigrams to capture phrases
print("\nExtracting features...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=2000,
    ngram_range=(1, 2),  # Use unigrams and bigrams
    min_df=3,            # Term must appear in at least 3 documents
    max_df=0.85          # Ignore terms that appear in more than 85% of documents
)
X = tfidf_vectorizer.fit_transform(cleaned_titles)

# Print some of the features
feature_names = tfidf_vectorizer.get_feature_names_out()
print(f"Number of features extracted: {len(feature_names)}")
print(f"Sample features: {', '.join(feature_names[:20])}...")

# Apply topic modeling first to learn major themes
print("\nDiscovering topics...")
n_topics = 12
lda = LatentDirichletAllocation(
    n_components=n_topics,
    max_iter=10,
    learning_method='online',
    random_state=42,
    n_jobs=-1
)
topic_distributions = lda.fit_transform(X)

# Print the top words for each discovered topic
print("\nTop terms per topic:")
for topic_idx, topic in enumerate(lda.components_):
    top_features_idx = topic.argsort()[:-11:-1]  # Get indices of top 10 words
    top_features = [feature_names[i] for i in top_features_idx]
    print(f"Topic #{topic_idx}: {', '.join(top_features)}")

# Normalize the topic distributions for better clustering
normalizer = Normalizer()
X_topics_normalized = normalizer.fit_transform(topic_distributions)

# Use KMeans clustering on the topic distributions (more compatible than AgglomerativeClustering)
print("\nClustering based on topic distributions...")
n_clusters = 8  # Adjust as needed
kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    n_init=10
)
clusters = kmeans.fit_predict(X_topics_normalized)

# Create DataFrame with clustering results
df = pd.DataFrame({
    "Original_Title": original_titles,
    "Cleaned_Title": cleaned_titles,
    "Topic_Vector": list(topic_distributions),
    "Cluster": clusters
})

# Analyze clusters
print("\n=== Cluster Analysis ===")
cluster_sizes = df["Cluster"].value_counts().sort_index()
for cluster_id, size in cluster_sizes.items():
    print(f"\nCluster {cluster_id} ({size} videos):")
    
    # Get all titles in this cluster
    cluster_titles = df[df["Cluster"] == cluster_id]["Cleaned_Title"].tolist()
    
    # Extract most common words
    all_words = " ".join(cluster_titles).split()
    most_common = Counter(all_words).most_common(10)
    
    print(f"Top words: {', '.join([word for word, count in most_common])}")
    
    # Print some examples
    print("Examples:")
    for title in df[df["Cluster"] == cluster_id]["Original_Title"].head(3).tolist():
        print(f"  • {title}")

# Generate a name for each cluster based on top words
cluster_names = {}
for cluster_id in range(n_clusters):
    cluster_titles = df[df["Cluster"] == cluster_id]["Cleaned_Title"].tolist()
    
    # Skip empty clusters
    if not cluster_titles:
        cluster_names[cluster_id] = f"Empty_Cluster_{cluster_id}"
        continue
    
    # Get most common words
    all_words = " ".join(cluster_titles).split()
    most_common = Counter(all_words).most_common(5)
    
    # Create a name from top 3 words
    name = "_".join([word for word, count in most_common[:3]])
    cluster_names[cluster_id] = name

# Add cluster names to DataFrame
df["Cluster_Name"] = df["Cluster"].map(cluster_names)

# Find similar videos within each cluster
print("\nFinding similar videos within clusters...")
for cluster_id in range(n_clusters):
    cluster_df = df[df["Cluster"] == cluster_id]
    if len(cluster_df) <= 1:
        continue
        
    # Convert topic vectors to numpy array
    vectors = np.array([np.array(v) for v in cluster_df["Topic_Vector"]])
    
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(vectors)
    
    # Find top 3 most similar pairs
    indices = np.unravel_index(np.argsort(similarity_matrix.flatten())[-4:-1], similarity_matrix.shape)
    pairs = list(zip(indices[0], indices[1]))
    
    print(f"\nCluster {cluster_id} - Most similar video pairs:")
    for i, j in pairs:
        if i != j:  # Skip self-comparisons
            title_i = cluster_df.iloc[i]["Original_Title"]
            title_j = cluster_df.iloc[j]["Original_Title"]
            sim_score = similarity_matrix[i, j]
            print(f"Similarity: {sim_score:.2f}")
            print(f"  • {title_i}")
            print(f"  • {title_j}")

# Optional: Visualize the clusters using t-SNE
print("\nVisualizing clusters with t-SNE...")
try:
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_topics_normalized)
    
    # Create a DataFrame for visualization
    viz_df = pd.DataFrame({
        "x": X_tsne[:, 0],
        "y": X_tsne[:, 1],
        "cluster": clusters,
        "title": original_titles
    })
    
    # Generate a scatter plot
    plt.figure(figsize=(12, 8))
    for cluster_id in range(n_clusters):
        subset = viz_df[viz_df["cluster"] == cluster_id]
        plt.scatter(subset["x"], subset["y"], label=f"Cluster {cluster_id}")
    
    plt.legend()
    plt.title("t-SNE Visualization of YouTube Video Clusters")
    
    # Save the visualization
    plt.savefig("ClustersDataFrame/cluster_visualization.png")
    print("Visualization saved as 'ClustersDataFrame/cluster_visualization.png'")
except Exception as e:
    print(f"Visualization failed: {e}")

# Save results
print("\nSaving cluster data...")
for cluster_id in range(n_clusters):
    cluster_df = df[df["Cluster"] == cluster_id]
    
    # Create filename with cluster name
    name = cluster_names[cluster_id]
    file_name = f"ClustersDataFrame/cluster_{cluster_id}_{name}.csv"
    
    # Save to CSV
    cluster_df.to_csv(file_name, index=False)
    print(f"Saved: {file_name}")

# Save complete results
df.to_csv("ClustersDataFrame/all_clusters.csv", index=False)

print("\n✅ Clustering complete! Results saved to 'ClustersDataFrame' folder")

# import json
# import pandas as pd
# import re
# import os
# from gensim.models import Word2Vec
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# from collections import Counter

# # Create a folder to store cluster CSVs
# os.makedirs("ClustersDataFrame", exist_ok=True)

# # Load JSON file
# path = "watch-history.json"
# with open(path, "r", encoding="utf-8") as f:
#     data = json.load(f)

# # Define custom stopwords (remove generic words like "best", "funny", "moments")
# custom_stopwords = {"best", "funny", "moments", "top", "new", "great", "ultimate", "official", "trailer"}

# titles = []
# for entry in data:
#     if entry.get("header") == "Youtube TV":  # Ignore YouTube TV
#         continue

#     title = entry.get("title", "").replace("Watched ", "").strip()
    
#     if "details" in entry and any(d.get("name") == "From Google Ads" for d in entry["details"]):
#         continue

#     if title.startswith("https://") or title == "":
#         continue

#     # Remove special characters and stopwords
#     title_cleaned = " ".join([word.lower() for word in re.sub(r"[^a-zA-Z0-9 ]", "", title).split() if word not in custom_stopwords])
    
#     titles.append(title_cleaned)

# print(f"Total valid videos: {len(titles)}")
# print("Sample Titles:", titles[:10])

# # Tokenize titles into lists of words
# titles_tokenized = [title.split() for title in titles]

# # Train Word2Vec model
# word2vec_model = Word2Vec(sentences=titles_tokenized, vector_size=100, window=5, min_count=1, workers=4)

# # Convert each title into an averaged word vector
# title_vectors = []
# for title in titles_tokenized:
#     vectors = [word2vec_model.wv[word] for word in title if word in word2vec_model.wv]
#     avg_vector = sum(vectors) / len(vectors) if vectors else [0] * 100  # Handle empty cases
#     title_vectors.append(avg_vector)

# # Normalize data before clustering
# scaler = StandardScaler()
# X = scaler.fit_transform(title_vectors)

# # Apply KMeans with more clusters
# num_clusters = 10  # Increase clusters for better separation
# kmeans = KMeans(n_clusters=num_clusters, random_state=42, init="k-means++", n_init=10)
# clusters = kmeans.fit_predict(X)

# # Store results in a DataFrame
# dataframe = pd.DataFrame({"Title": titles, "Cluster": clusters})

# # Auto-label clusters using most common words
# cluster_labels = {}

# for cluster_num in range(num_clusters):
#     cluster_titles = dataframe[dataframe["Cluster"] == cluster_num]["Title"].tolist()
    
#     words = [word for title in cluster_titles for word in title.split()]
#     most_common_words = Counter(words).most_common(5)  # Get top words in cluster
    
#     cluster_name = " / ".join([word for word, _ in most_common_words])  # Assign category name
#     cluster_labels[cluster_num] = cluster_name

# # Add labeled clusters to DataFrame
# dataframe["Cluster Name"] = dataframe["Cluster"].map(cluster_labels)

# # Print cluster categories
# print("\n### Cluster Names ###\n")
# for cluster, label in cluster_labels.items():
#     print(f"Cluster {cluster} → {label}")

# # Save each cluster separately
# for cluster_num in range(num_clusters):
#     file_name = f"ClustersDataFrame/cluster_{cluster_num}.csv"
#     df = dataframe[dataframe["Cluster"] == cluster_num]
#     df.to_csv(file_name, index=False)

# print("\n✅ Clusters saved in 'ClustersDataFrame' folder")


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