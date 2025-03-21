# 📺 YouTube Watch History Clustering Report
### Paul Garces - YouTube-NLP-Analysis

This project analyzes and clusters YouTube video titles from your personal watch history using Natural Language Processing (NLP) techniques. The goal is to automatically group similar types of videos—such as music videos, sports highlights, and apartment tours—into thematic clusters based on title content.

---

## 🔍 Project Overview

- **Data Source:** `watch-history.json` from Google Takeout
- **Tooling:** Python, Pandas, Gensim (Word2Vec), scikit-learn (KMeans), Regex
- **Goal:** Cluster 31,000+ video titles into meaningful groups based on shared content or topic

---

## 🧠 Methodology

### 🧼 1. Data Cleaning
- Removed irrelevant records such as:
  - YouTube TV entries
  - Sponsored ads (`"From Google Ads"`)
  - URLs or empty titles
- Removed common stopwords (e.g., *"the"*, *"to"*, *"with"*) to focus on meaningful terms

---

### 🎯 2. Keyword Boosting
To help cluster certain niche categories better, a **keyword boosting** method was used:
- If a title contains **2 or more** keywords from a category, that category's label is repeated 3 times in the title for extra weight.

| Category      | Triggered By Keywords                                  |
|---------------|--------------------------------------------------------|
| `MUSICCATEGORY`     | music, audio, ft, video, lyrics, remix, live         |
| `SPORTSCATEGORY`    | highlights, league, nba, soccer, football, sports    |
| `APARTMENTCATEGORY` | apartment, tour, nyc, penthouse, house, manhattan    |

---

### 💬 3. Word Embeddings (Word2Vec)
- Tokenized all boosted titles into word lists
- Trained a **Word2Vec** model to represent each title as the **average vector of its words**

---

### 📊 4. Clustering (KMeans)
- Normalized vectors using `StandardScaler`
- Applied **KMeans clustering** with `n_clusters = 6` to group similar titles
- Each cluster was labeled using the **top 5 most frequent words** found in the cleaned titles of that group

---

## 📁 Output

Each cluster is saved in its own CSV file inside the `ClustersDataFrame/` folder.

For each file:
- `Original_Title`: Raw YouTube title
- `Cleaned_Title`: Title after cleaning and stopword removal
- `Cluster`: Numerical cluster ID
- `Cluster_Name`: Label derived from most common words in the group

---