# Paul Garces - YouTube Watch History Analysis
# This .py script reads the watch history from a JSON file and performs clustering on the video titles
# IMPORTANT: Rerunning this script will result in different clustering outcomes!!!!

# downloading/importing the libraries that are needed
import json
import pandas as pd
import re
import os
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter

#creating the folders needed to actually put the clusters into as dataframes
os.makedirs("ClustersDataFrame", exist_ok=True)

# this part is...
path = "watch-history.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
# just loading in the actual json file that has the youtube watch history titles and data

# this list of words are those common words that don't really add any meaning to the clustering, and as said, show up A LOT
# so they're removed so that we can actually get those important words in the titles
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

# these lists include words that are related to the specific content that will in a way help indicate or point a video title
# to a certain category/cluster. basically the purpose of these is to help identifty clusters more effectively
# these will be used in the boosting section of the code which is pretty important
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

# just starting an empty list to store the original video titles, the cleaned titles and then the boosted titles
original_titles = []
cleaned_titles = []
boosted_titles = []

# this little section of the code is...
for entry in data:
    if entry.get("header") == "Youtube TV":
        continue
    title = entry.get("title", "").replace("Watched ", "").strip()
    if "details" in entry and any(d.get("name") == "From Google Ads" for d in entry["details"]):
        continue
    if title.startswith("https://") or title == "":
        continue
 # getting rid of data that were watched on Youtube TV since those were mainly live things that were going and aren't too meaningful
 # it's also getting rid of "videos" that came from ads since they're adds and they're not actually and truly part of the youtube video itself
 # also cleaning video titles that start with "watched: "since the data in json file video titles starts with those for videos that were watched
 # also skips videos that have urls and/or no titles at all

# this is keeping the original title for later which will be used in the actual output of the clustering dataframe
    original_titles.append(title)
# this is cleaning the title and getting rid of punctuation, then puts it into lowercase, and then removes words from the custom_stopwords list    
    title_cleaned = " ".join([word.lower() for word in re.sub(r"[^a-zA-Z0-9 ]", "", title).split()
                              if word.lower() not in custom_stopwords])
    cleaned_titles.append(title_cleaned) # adding it to the empty cleaned_titles list
    
    title_lower = title.lower()

# this whole section is the actual boosting based on themes which will use the keywords from above
    boosted_title = title_cleaned # using the cleaned title as the base for this process
    
    # for example, in the music case, we're getting the sum/total of the number of times that a word from the music_keywords list appears in the 
    # lower cased version of our video title.
    # if there are two or more keywords in the title, then we add MUSICCATEGORY x3 to the title, which helps...
    # word2vec "know" that the title is strongly related to music (this will also help with further clustering)
    # same thing goes for the other two parts after. these are also known as category markers
    music_count = sum(1 for keyword in music_keywords if keyword in title_lower)
    if music_count >= 2:
        boosted_title += " MUSICCATEGORY MUSICCATEGORY MUSICCATEGORY"
    
    sports_count = sum(1 for keyword in sports_keywords if keyword in title_lower)
    if sports_count >= 2:
        boosted_title += " SPORTSCATEGORY SPORTSCATEGORY SPORTSCATEGORY"
    
    apartment_count = sum(1 for keyword in apartment_keywords if keyword in title_lower)
    if apartment_count >= 2:
        boosted_title += " APARTMENTCATEGORY APARTMENTCATEGORY APARTMENTCATEGORY"
    
    boosted_titles.append(boosted_title) # after the process of finding keywords in titles is done, if two or more keywords found in title...
    # the new title in the boosted_title would look like "VIDEO TITLE MUSICCATEGORY MUSICCATEGORY MUSICCATEGORY"
    # if the titles aren't boosted, they're still stored here for the later vectorization process
    # both boosted un-boosted titles will follow the same process/pipeline below

print(f"Total valid videos: {len(original_titles)}")

# here each title is split into a list which will help word2vec process each word seperately 
titles_tokenized = [title.split() for title in boosted_titles]

# here the word2vec model is being created where the titles are the input, there is a vector size of 100 meaning that each word is
# represented as a 100 dimensional vector, the window of 5 is the max distance between predicted and current word in sentence for training,
# words must appear once to be considered in the model, and the number of worker threads if four to train the model
word2vec_model = Word2Vec(sentences=titles_tokenized, vector_size=100, window=5, min_count=1, workers=4)

# in this section...
title_vectors = []
for title in titles_tokenized:
    vectors = [word2vec_model.wv[word] for word in title if word in word2vec_model.wv]
    avg_vector = sum(vectors) / len(vectors) if vectors else [0] * 100
    title_vectors.append(avg_vector)
# we're extracting word2vec vectors for its words and computing the average vector. If there's no words
# in the word2vec vocab, then we're using a vector of 0

# here we're normalizing the title vectors with standard scaling which is where we subtract the mean and sacling the "unit varianced"
# this is pretty standard in all ML models and preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(title_vectors)

# now actually applying Kmeans to the normalized vectors to create 6 clusters, where the titles are now grouped
# based on how similar the vectors are. 
num_clusters = 6
kmeans = KMeans(n_clusters=num_clusters, random_state=42, init="k-means++", n_init=10)
clusters = kmeans.fit_predict(X)

# creating a dataframe with the original title, the cleaned title, and then which number cluster number it was assigned to
dataframe = pd.DataFrame({
    "Original_Title": original_titles,
    "Cleaned_Title": cleaned_titles,
    "Cluster": clusters
})

# here we're just getting the most common name in each cluster so we can get a sense of what topics the videos are about or related to and grouped by
cluster_labels = {}
for cluster_num in range(num_clusters):
    cluster_titles = dataframe[dataframe["Cluster"] == cluster_num]["Cleaned_Title"].tolist()
    words = [word for title in cluster_titles for word in title.split()]
    most_common_words = Counter(words).most_common(5)
    cluster_name = " / ".join([word for word, count in most_common_words])
    cluster_labels[cluster_num] = cluster_name

dataframe["Cluster_Name"] = dataframe["Cluster"].map(cluster_labels)

# the rest of this is just creating the csv's and exporting them which can be found in the ClusterDataFrame folder
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