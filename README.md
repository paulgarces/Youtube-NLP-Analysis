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

Each cluster, a total of six, is saved in its own CSV file inside the `ClustersDataFrame/` folder.

- Cluster 0: `GlobalSportsCoverage.csv`
- Cluster 1: `MusicVideosLyrics.csv`
- Cluster 2: `GamingChallenges.csv`
- Cluster 3: `SoccerSportHighlights.csv`
- Cluster 4: `HousingRealEstate.csv`
- Cluster 5: `MixedMedia.csv`

For each file:
- `Original_Title`: Raw YouTube title
- `Cleaned_Title`: Title after cleaning and stopword removal
- `Cluster`: Numerical cluster ID
- `Cluster_Name`: Most common words in the group

---

## 📑 Cluster Overview

### 🏆 Cluster 0: Global Sports Coverage

This cluster contains content mainly related to international and domestic sports. It includes a mix of official match highlights, memorable moments, player-centric content, trivia, and challenges. Videos frequently reference global tournaments, popular teams, and famous athletes, as well as educational and entertainment-focused sports content.

**Top Keywords:** `football`, `highlights`, `2022`,`college` ,`game`, 

**Example Titles:**
- *£500 Jabulani v £150 Brazuca v £100 Telstar | World Cup Ball Battle*  
- *HE WENT 10/10! PERFECT FOOTBALL KNOWLEDGE ON SHOW 👏 #shorts*  
- *Michigan State vs. USC - First Round NCAA tournament extended highlights*  
- *All Sports Golf Battle 3 | Dude Perfect*  
- *Greatest American Sports Moments of the Decade [2010-2019]*

## 🎵 Cluster 1: Music Videos & Lyrics

This cluster primarily consists of music-related content**, including official music videos, lyric videos, live performances, and remixes. It features a variety of genres, artists, and collaborations, often tagged with terms like "official video", "ft." (featuring), "lyrics", and "remix". The content also includes trending songs, new releases, and live concerts.

**Top Keywords:** `video`, `ft`, `music`, `audio`, `feat`

**Example Titles:**
- *Adele - Someone Like You (Official Music Video)*
- *PUBLIC - Make You Mine (Official Lyric Video)*
- *Mac Miller - My Favorite Part (feat. Ariana Grande) (Live)*
- *The Weeknd ft. Ariana Grande - Save Your Tears (Remix)*
- *Rainbow Bap (Remix) (Official Visualizer)*

## 🎮 Cluster 2: Gaming & Challenges

This cluster consists of gaming-related content, including gameplay highlights, challenges, esports competitions, and gaming-related discussions. The content spans across different gaming genres, featuring popular games, player-versus-player (PvP) matches, and reaction-based gaming moments. Many of these videos include challenges, speedruns, or collaborations between popular gaming creators.

**Top Keywords:** `sidemen`, `ksi`, `office`, `w2s`, `harry`

**Example Titles:**
- *SIDEMEN UNO CHAOS MODE*
- *I made Marvel Rivals Game of the Year worthy*
- *I Played Knock Off's of the most FAMOUS GAMES! (GTA, COD, MINECRAFT, FORTNITE)*
- *100 Players Simulate THE HUNGER GAMES in Minecraft...v*
- *Batman Arkham Origins is so much better than I remember*

## ⚽ Cluster 3: Soccer & Sports Highlights

This cluster is primarily focused on soccer match highlights, with extensive coverage of Premier League games, UEFA Champions League matches, and international tournaments. It includes full-match recaps, best goals, key player performances, and post-game analysis.

While the majority of content revolves around football (soccer), this cluster also contains highlights from other sports, such as college athletics, proffesional athletics, and key moments from major sporting events. However, soccer remains the dominant focus, making this cluster particularly useful for fans of club and international football.

**Top Keywords:** `highlights`, `sports`, `league`, `v`, `united`

**Example Titles:**
- *Manchester United vs. Galatasaray: Extended Highlights | UCL Group Stage MD 2 | CBS Sports Golazo*
- *#1 LSU vs Texas Highlights (Great Game!) | 2023 College Baseball Highlights*
- *Germany vs Hungary | Highlights | UEFA Nations League*
- *Top 23 Premier League goals of 2023 | NBC Sports*
- *Denver Broncos vs. Seattle Seahawks | NFL 2024 Week 1 Game Highlights*

## 🏡 Cluster 4: Housing & Real Estate

This cluster primarily consists of content related to housing, real estate, and apartment tours. Many of these videos feature apartment tours in major metropolitan areas such as New York City (NYC) and Los Angeles (LA), and Philadelphia, highlighting modern interiors, pricing comparisons, and unique architectural designs.

**Top Keywords:** `nyc`, `apartment`, `tour`, `la`, `penthouse`

**Example Titles:**
- *Inside a $75,000,000 OFF MARKET NYC Penthouse | Mansion Tour	*
- *i toured 14 philly apartments and this is how it went.*
- *4 Million LA Apartment #shorts #la*
- *Day in the life in my new NYC apartment!*
- *How SoHo NYC Became The Cast Iron District | Walking Tour | Architectural Digest*

## 🎬 Cluster 5: Mixed Media & Entertainment Shorts

This cluster includes a broad range of entertainment content. It features a mix of viral videos, technology advancements, gaming highlights, TV show and movie snippets and moments, comedy skits, and reaction-based content. 

**Top Keywords:** `de`, `why`, `sidemen`, `ever`, `out`

**Example Titles:**
- *How THANOS bullied the INFINTY STONE from THE AVENGERS in Wakanda*
- *Best LEGO alt Build! #lego #legospeedchampions #legocars #legobuild	*
- *Loyalty is a two-way street, if I ask you for it, you will receive it from me #suits #shorts*
- *Tesla Optimus Robot Showcases Advanced Walking on Uneven Terrain*
- *SIDEMEN REACT TO CONSPIRACY THEORIES THAT BECAME TRUE*