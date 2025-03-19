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
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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
    titles.append(title)  # Keep original title for readability
    
cleaned_titles = [" ".join([word.lower() for word in re.sub(r"[^a-zA-Z0-9 ]", "", title).split() if word not in custom_stopwords]) for title in titles]

print(f"Total valid videos: {len(titles)}")
print("Sample Titles:", titles[:5])

# Define predetermined categories with keywords
predefined_categories = {
    "Music": ["song", "music", "dance", "choreography", "album", "concert", "lyrics", "remix", "beat", "instrumental", 
              "singing", "vocalist", "band", "rap", "hip hop", "jazz", "rock", "pop", "edm", "dj", "official audio", 
              "official music video", "mv", "cover", "live performance", "festival", "choir", "orchestra", "piano", "guitar", 
              "drum", "bass", "vocalist", "mashup", "collab", "feat", "ft.", "featuring", "producer", "track", "mixtape", 
              "lofi", "lo-fi", "playlist", "acoustic", "unplugged", "anthem", "viral song", "tiktok song", "Billboard", 
              "Grammy", "chord", "note", "rhythm", "melody", "tune", "ballad", "kpop", "k-pop", "jpop", "j-pop", 
              "phonk", "slowed", "reverb", "extended", "mix", "trap", "club", "dance music", "electronic"],
              
    "Gaming": ["game", "gaming", "playthrough", "gameplay", "multiplayer", "speedrun", "let's play", "minecraft", "fortnite", 
               "valorant", "league of legends", "lol", "dota", "strategy", "rpg", "fps", "walkthrough", "console", "pc game", 
               "xbox", "playstation", "ps4", "ps5", "nintendo", "switch", "steam", "esports", "streamer", "twitch", "dlc", 
               "mod", "modded", "cheat", "glitch", "easter egg", "achievement", "tutorial", "tips", "tricks", "guide", 
               "boss fight", "raid", "quest", "mission", "level", "character build", "class", "hero", "champion", "weapon", 
               "loadout", "battle royale", "pvp", "pve", "mmorpg", "moba", "rts", "sim", "simulator", "sandbox", "open world", 
               "survival", "craft", "building", "sidemen", "ksi", "search and destroy", "call of duty", "cod", "warzone", 
               "apex legends", "gta", "grand theft auto", "overwatch", "among us", "roblox", "pubg", "tekken", "street fighter", 
               "mortal kombat", "fighting game", "smash bros", "super smash", "pokemon", "zelda", "mario", "sonic", "halo", 
               "destiny", "world of warcraft", "wow", "sims", "animal crossing", "rocket league", "fall guys", "indie game", 
               "speedrunning", "no hit", "no damage", "world record", "high score", "challenge", "hardcore", "permadeath",
               "nuzlocke", "stream highlight", "rage", "funny moments", "fails", "wins"],
               
    "Sports": ["football", "basketball", "soccer", "nba", "nfl", "mlb", "fifa", "match", "highlights", "sport", "athletic", 
               "champion", "tournament", "boxing", "cricket", "tennis", "golf", "olympics", "world cup", "ucl", "champions league", 
               "premier league", "la liga", "bundesliga", "serie a", "ligue 1", "mls", "goal", "touchdown", "dunk", "shot", "pass", 
               "tackle", "save", "header", "assist", "free kick", "penalty", "red card", "yellow card", "var", "referee", "umpire", 
               "player", "team", "coach", "manager", "transfer", "signing", "contract", "draft", "trade", "stadium", "arena", 
               "field", "pitch", "court", "ring", "gym", "fitness", "workout", "training", "exercise", "routine", "strength", 
               "cardio", "muscle", "weight lifting", "crossfit", "yoga", "pilates", "marathon", "race", "run", "sprint", 
               "swimming", "cycling", "biking", "skateboarding", "snowboarding", "skiing", "surfing", "martial arts", "mma", 
               "ufc", "wrestling", "wwe", "formula 1", "f1", "nascar", "racing", "volleyball", "hockey", "nhl", "baseball", 
               "rugby", "bowling", "billiards", "snooker", "darts", "table tennis", "badminton", "analysis", "tactics", 
               "stats", "statistics", "top 10", "ranking", "all time", "classic", "legendary", "iconic", "greatest", 
               "comeback", "upset", "underdog", "playoff", "final", "semifinal", "quarterfinal", "knockout", "group stage"],
               
    "Movies & TV": ["movie", "film", "series", "episode", "season", "tv show", "cinema", "documentary", "netflix", "hulu", 
                    "disney", "actor", "director", "plot", "review", "trailer", "scene", "character", "production", "hbo", 
                    "amazon prime", "peacock", "paramount", "universal", "warner bros", "sony pictures", "pixar", "marvel", 
                    "dc", "star wars", "harry potter", "lord of the rings", "game of thrones", "stranger things", "breaking bad", 
                    "office", "friends", "simpsons", "family guy", "south park", "anime", "manga", "cartoon", "animation", 
                    "studio ghibli", "disney", "horror", "thriller", "action", "adventure", "sci-fi", "fantasy", "romance", 
                    "drama", "comedy", "mystery", "crime", "western", "musical", "biography", "history", "war", "oscar", 
                    "emmy", "award", "nomination", "box office", "rating", "critics", "audience", "behind the scenes", 
                    "bloopers", "deleted scenes", "extended cut", "director's cut", "remake", "reboot", "sequel", "prequel", 
                    "spinoff", "franchise", "universe", "adaptation", "based on", "screenplay", "script", "cast", "acted", 
                    "performance", "cinematography", "visual effects", "cgi", "soundtrack", "score", "theme song", "intro", 
                    "outro", "opening", "ending", "cliffhanger", "pilot", "finale", "midseason", "binge watch", "recap", 
                    "explained", "theory", "easter eggs", "cameo", "crossover", "interview", "press tour", "convention", 
                    "panel", "q&a", "fan", "fandom"],
                    
    "Comedy": ["comedy", "funny", "joke", "stand up", "humor", "prank", "parody", "sketch", "comedian", "laugh", "meme", 
               "roast", "sitcom", "stand-up", "improv", "improvisation", "satire", "skit", "blooper", "outtake", "gag", 
               "bloopers", "fail", "fails", "compilation", "stand up comedy", "comedy central", "saturday night live", 
               "snl", "late night", "talk show", "podcast", "impression", "impersonation", "mockumentary", "spoof", 
               "slapstick", "dark humor", "dry humor", "wit", "wordplay", "pun", "dad joke", "inside joke", "viral", 
               "troll", "trolling", "reaction", "try not to laugh", "whose line", "comedy club", "comedy special", 
               "comedy tour", "comedy festival", "comedy show", "comedy movie", "comedy series", "funny moments", 
               "funny video", "funny clip", "funny scene", "funny interview", "funny commercial", "funny ad", "comedian", 
               "comedic", "hilarious", "lmao", "lol", "rofl", "standup", "comedy skit"],
               
    "Education": ["tutorial", "learn", "education", "course", "lecture", "professor", "school", "university", "college", 
                  "teach", "training", "exam", "lesson", "study", "explanation", "guide", "how to", "educational", 
                  "academic", "scholar", "research", "science", "math", "mathematics", "physics", "chemistry", "biology", 
                  "history", "geography", "literature", "language", "grammar", "vocabulary", "spelling", "writing", 
                  "reading", "arithmetic", "algebra", "calculus", "geometry", "statistics", "economics", "finance", 
                  "accounting", "business", "management", "marketing", "psychology", "sociology", "anthropology", 
                  "philosophy", "ethics", "law", "medicine", "engineering", "architecture", "design", "art", "music theory", 
                  "theory", "concept", "principle", "method", "technique", "skill", "practice", "exercise", "worksheet", 
                  "quiz", "test", "assignment", "homework", "project", "experiment", "demonstration", "example", "sample", 
                  "solution", "answer", "problem", "question", "inquiry", "investigation", "discovery", "research", "thesis", 
                  "dissertation", "paper", "journal", "publication", "textbook", "handbook", "manual", "reference", "citation", 
                  "bibliography", "glossary", "vocabulary", "term", "definition", "meaning", "interpretation", "analysis", 
                  "synthesis", "evaluation", "assessment", "feedback", "review", "revision", "edit", "correction", "improvement", 
                  "development", "progress", "advancement", "achievement", "success", "expertise", "mastery", "proficiency", 
                  "competence", "capability", "ability", "knowledge", "wisdom", "insight", "understanding", "comprehension", 
                  "learning", "student", "teacher", "instructor", "mentor", "tutor", "coach", "advisor", "guide", "facilitator", 
                  "classroom", "lecture hall", "laboratory", "library", "archive", "database", "resource", "material", "curriculum", 
                  "syllabus", "program", "degree", "certificate", "diploma", "credential", "qualification", "requirement", 
                  "prerequisite", "corequisite", "ted talk", "tedx", "educational channel", "crash course", "khan academy"],
                  
    "Technology": ["tech", "review", "unboxing", "gadget", "smartphone", "computer", "laptop", "software", "hardware", 
                   "programming", "coding", "development", "ai", "machine learning", "data science", "algorithm", "technology", 
                   "device", "electronics", "digital", "robotics", "automation", "iot", "internet of things", "smart home", 
                   "smart device", "virtual reality", "vr", "augmented reality", "ar", "mixed reality", "mr", "metaverse", 
                   "blockchain", "cryptocurrency", "bitcoin", "ethereum", "nft", "web3", "cloud", "server", "database", 
                   "network", "wireless", "wifi", "bluetooth", "5g", "4g", "lte", "broadband", "fiber", "satellite", 
                   "cable", "dsl", "modem", "router", "switch", "hub", "firewall", "vpn", "proxy", "dns", "ip", "protocol", 
                   "standard", "specification", "api", "sdk", "framework", "library", "package", "module", "component", 
                   "architecture", "design pattern", "object oriented", "functional", "procedural", "imperative", "declarative", 
                   "scripting", "markup", "stylesheet", "front end", "back end", "full stack", "web development", "mobile development", 
                   "app development", "game development", "devops", "ci/cd", "testing", "qa", "debug", "bug", "fix", "patch", 
                   "update", "upgrade", "version", "release", "alpha", "beta", "rc", "stable", "lts", "eol", "deprecated", 
                   "legacy", "compatibility", "interoperability", "integration", "migration", "conversion", "transformation", 
                   "optimization", "performance", "scalability", "reliability", "availability", "security", "privacy", "encryption", 
                   "decryption", "hashing", "authentication", "authorization", "access control", "identity management", "biometrics", 
                   "facial recognition", "fingerprint", "voice recognition", "gesture control", "touch", "haptic", "feedback", 
                   "input", "output", "display", "monitor", "screen", "resolution", "refresh rate", "fps", "hdr", "sdr", "color", 
                   "brightness", "contrast", "gamma", "saturation", "hue", "pixel", "subpixel", "oled", "qled", "lcd", "led", 
                   "projector", "speaker", "microphone", "camera", "webcam", "sensor", "accelerometer", "gyroscope", "magnetometer", 
                   "gps", "location", "navigation", "mapping", "tracking", "surveillance", "drone", "robot", "autonomous", 
                   "self driving", "vehicle", "electric", "battery", "charging", "solar", "renewable", "green", "eco friendly", 
                   "sustainable", "energy efficient", "power consumption", "heat", "cooling", "fan", "heatsink", "thermal", 
                   "overclocking", "undervolting", "benchmark", "stress test", "load test", "synthetic test", "real world test", 
                   "comparison", "versus", "vs", "alternative", "competitor", "market", "industry", "sector", "startup", 
                   "venture", "investment", "funding", "acquisition", "merger", "ipo", "stock", "share", "profit", "revenue", 
                   "cost", "price", "value", "warranty", "guarantee", "service", "support", "repair", "maintenance", "care", 
                   "technical specifications", "specs", "dimension", "weight", "size", "form factor", "ergonomics", "usability", 
                   "user experience", "ux", "user interface", "ui", "gui", "cli", "tui", "hci", "interaction", "gesture", 
                   "multitouch", "swipe", "pinch", "zoom", "scroll", "drag", "drop", "click", "tap", "press", "hold", 
                   "keyboard", "mouse", "trackpad", "stylus", "pen", "remote", "controller", "joystick", "wheel", "pedal", 
                   "button", "switch", "dial", "slider", "knob", "peripheral", "accessory", "adapter", "converter", "hub", 
                   "dock", "station", "stand", "mount", "bracket", "case", "cover", "skin", "protector", "film", "glass", 
                   "tech news", "tech blog", "tech channel", "tech company"],
                   
    "Travel & Lifestyle": ["travel", "vlog", "lifestyle", "tour", "apartment", "house", "home", "mansion", "penthouse", 
                           "renovation", "decoration", "design", "trip", "journey", "vacation", "hotel", "resort", "city", 
                           "country", "destination", "tourist", "tourism", "sight", "sightseeing", "attraction", "landmark", 
                           "monument", "museum", "gallery", "park", "garden", "beach", "mountain", "hill", "valley", "canyon", 
                           "desert", "forest", "jungle", "island", "lake", "river", "ocean", "sea", "waterfall", "hot spring", 
                           "geyser", "volcano", "cave", "national park", "reserve", "wildlife", "safari", "zoo", "aquarium", 
                           "theme park", "amusement park", "water park", "adventure", "expedition", "exploration", "discovery", 
                           "backpacking", "camping", "hiking", "trekking", "climbing", "mountaineering", "cycling", "biking", 
                           "road trip", "cruise", "sailing", "boating", "flight", "airplane", "airport", "train", "railway", 
                           "station", "bus", "subway", "metro", "tram", "cab", "taxi", "ride share", "uber", "lyft", "rental", 
                           "luggage", "baggage", "packing", "unpacking", "check in", "check out", "reservation", "booking", 
                           "itinerary", "schedule", "map", "guide", "guidebook", "brochure", "pamphlet", "information", 
                           "visitor center", "souvenir", "gift", "memento", "photo", "photography", "video", "documentary", 
                           "diary", "journal", "blog", "vlog", "review", "recommendation", "tip", "advice", "suggestion", 
                           "warning", "caution", "safety", "security", "emergency", "first aid", "health", "insurance", 
                           "visa", "passport", "id", "identification", "document", "customs", "immigration", "border", 
                           "international", "domestic", "local", "native", "resident", "expatriate", "digital nomad", 
                           "remote work", "workation", "bleisure", "business trip", "conference", "meeting", "event", 
                           "festival", "celebration", "holiday", "vacation", "getaway", "weekend", "day trip", "tour guide", 
                           "escort", "companion", "group", "solo", "family", "couple", "honeymoon", "anniversary", "birthday", 
                           "graduation", "retirement", "relocation", "moving", "settling", "living abroad", "immigration", 
                           "emigration", "migration", "refugee", "asylum", "citizenship", "nationality", "identity", "culture", 
                           "tradition", "custom", "etiquette", "manner", "behavior", "attitude", "mindset", "perspective", 
                           "worldview", "philosophy", "religion", "spirituality", "belief", "faith", "ritual", "ceremony", 
                           "festival", "celebration", "party", "event", "gathering", "meetup", "convention", "conference", 
                           "summit", "forum", "symposium", "seminar", "workshop", "class", "course", "program", "retreat", 
                           "camp", "glamping", "luxury", "budget", "affordable", "expensive", "cheap", "free", "paid", 
                           "all inclusive", "package", "deal", "discount", "promotion", "sale", "offer", "seasonal", 
                           "off season", "peak season", "high season", "low season", "shoulder season", "weather", "climate", 
                           "temperature", "humidity", "precipitation", "rain", "snow", "sun", "wind", "storm", "hurricane", 
                           "typhoon", "tornado", "earthquake", "tsunami", "flood", "drought", "fire", "disaster", "emergency", 
                           "crisis", "situation", "condition", "circumstance", "decor", "interior design", "exterior design", 
                           "architecture", "landscaping", "gardening", "farming", "homesteading", "self sufficiency", 
                           "sustainability", "eco friendly", "green", "organic", "natural", "synthetic", "artificial", 
                           "processed", "raw", "whole", "healthy", "clean", "dirty", "messy", "organized", "neat", "tidy", 
                           "cluttered", "minimalist", "maximalist", "eclectic", "bohemian", "rustic", "vintage", "retro", 
                           "antique", "modern", "contemporary", "traditional", "classical", "victorian", "colonial", "industrial", 
                           "farmhouse", "cottage", "cabin", "apartment", "condo", "villa", "bungalow", "townhouse", "duplex", 
                           "split level", "ranch", "mcmansion", "estate", "palace", "castle", "fortress", "citadel", "tower", 
                           "skyscraper", "tiny house", "mobile home", "rv", "camper", "van life", "tiny living", "minimal living", 
                           "home tour", "house tour", "property tour", "real estate"],
                           
    "Food & Cooking": ["food", "recipe", "cooking", "chef", "kitchen", "baking", "restaurant", "cuisine", "meal", "dish", 
                       "ingredient", "taste", "flavor", "culinary", "diet", "nutrition", "breakfast", "brunch", "lunch", 
                       "dinner", "supper", "snack", "appetizer", "starter", "main course", "entree", "side dish", "dessert", 
                       "beverage", "drink", "cocktail", "mocktail", "smoothie", "juice", "tea", "coffee", "water", "milk", 
                       "soda", "pop", "beer", "wine", "spirit", "liquor", "alcohol", "meat", "poultry", "beef", "pork", 
                       "lamb", "chicken", "turkey", "duck", "fish", "seafood", "shellfish", "vegetable", "fruit", "grain", 
                       "rice", "pasta", "noodle", "bread", "pastry", "cake", "cookie", "pie", "tart", "pudding", "ice cream", 
                       "yogurt", "cheese", "dairy", "egg", "oil", "fat", "butter", "margarine", "sugar", "sweetener", "salt", 
                       "pepper", "spice", "herb", "seasoning", "condiment", "sauce", "dressing", "marinade", "glaze", "rub", 
                       "batter", "dough", "crust", "filling", "topping", "garnish", "presentation", "plating", "menu", 
                       "recipe book", "cookbook", "cook", "chef", "baker", "pastry chef", "patissier", "chocolatier", 
                       "butcher", "fishmonger", "greengrocer", "grocer", "farmer", "producer", "artisan", "handmade", 
                       "homemade", "from scratch", "store bought", "processed", "fresh", "frozen", "canned", "preserved", 
                       "pickled", "fermented", "smoked", "dried", "dehydrated", "cured", "raw", "rare", "medium rare", 
                       "medium", "medium well", "well done", "boiled", "simmered", "poached", "steamed", "sauteed", "stir fried", 
                       "deep fried", "pan fried", "roasted", "baked", "broiled", "grilled", "barbecued", "smoked", "braised", 
                       "stewed", "slow cooked", "pressure cooked", "microwave", "no cook", "cold", "hot", "warm", "room temperature", 
                       "refrigerated", "frozen", "thawed", "marinated", "seasoned", "spiced", "flavored", "sweet", "sour", 
                       "bitter", "salty", "umami", "spicy", "hot", "mild", "medium", "tangy", "tart", "acidic", "creamy", 
                       "smooth", "crunchy", "crispy", "tender", "tough", "chewy", "soft", "hard", "gooey", "sticky", "juicy", 
                       "dry", "moist", "wet", "thick", "thin", "chunky", "pureed", "blended", "whipped", "beaten", "folded", 
                       "kneaded", "rolled", "cut", "sliced", "diced", "chopped", "minced", "grated", "shredded", "zested", 
                       "peeled", "cored", "seeded", "pitted", "hulled", "shucked", "trimmed", "cleaned", "washed", "rinsed", 
                       "drained", "separated", "combined", "mixed", "incorporated", "emulsified", "restaurant", "cafe", 
                       "bistro", "diner", "drive in", "drive thru", "fast food", "takeout", "delivery", "catering", "buffet", 
                       "all you can eat", "prix fixe", "tasting menu", "chef's table", "pop up", "food truck", "street food", 
                       "food stall", "food court", "food hall", "grocery", "supermarket", "market", "farmers market", "csa", 
                       "farm to table", "organic", "local", "sustainable", "seasonal", "dietary", "nutritional", "calories", 
                       "fat", "carb", "protein", "fiber", "vitamin", "mineral", "supplement", "vegetarian", "vegan", "plant based", 
                       "pescatarian", "flexitarian", "omnivore", "carnivore", "herbivore", "raw food", "paleo", "keto", "low carb", 
                       "high protein", "low fat", "gluten free", "dairy free", "nut free", "soy free", "egg free", "kosher", 
                       "halal", "allergy", "intolerance", "sensitivity", "digestive", "gut health", "probiotic", "prebiotic", 
                       "fermented", "cultured", "food review", "restaurant review", "taste test", "food challenge", "mukbang", 
                       "asmr eating", "recipe tutorial", "cooking show", "baking championship", "food competition", "masterchef", 
                       "top chef", "food network", "tasty", "bon appetit", "epicurious", "babish", "matty matheson", "gordon ramsay", 
                       "jamie oliver", "nigella lawson", "bobby flay", "guy fieri", "ina garten", "alton brown", "binging with babish", 
                       "hot ones", "first we feast", "all recipes", "food wishes", "sorted food", "joshua weissman", "matty matheson", 
                       "sam the cooking guy", "kenji lopez alt", "adam ragusea", "ethan chlebowski", "j kenji lopez alt"],
                       
    "News & Politics": ["news", "politics", "analysis", "election", "debate", "president", "government", "campaign", "senator", 
                        "congress", "parliament", "candidate", "vote", "political", "policy", "issue", "current events", 
                        "breaking news", "latest news", "headline", "top story", "developing story", "update", "coverage", 
                        "report", "investigation", "expose", "documentary", "interview", "press conference", "statement", 
                        "announcement", "declaration", "proclamation", "address", "speech", "talk", "discussion", "conversation", 
                        "dialogue", "debate", "argument", "dispute", "controversy", "scandal", "affair", "incident", "event", 
                        "situation", "crisis", "emergency", "disaster", "catastrophe", "tragedy", "accident", "conflict", 
                        "war", "battle", "fight", "struggle", "revolution", "rebellion", "uprising", "riot", "protest", 
                        "demonstration", "rally", "march", "strike", "boycott", "movement", "campaign", "initiative", 
                        "program", "plan", "proposal", "bill", "law", "legislation", "regulation", "rule", "policy", 
                        "guideline", "directive", "mandate", "order", "command", "demand", "requirement", "obligation", 
                        "duty", "responsibility", "right", "freedom", "liberty", "justice", "equality", "equity", "fairness", 
                        "discrimination", "prejudice", "bias", "stereotype", "racism", "sexism", "homophobia", "transphobia", 
                        "xenophobia", "islamophobia", "antisemitism", "hate", "love", "peace", "harmony", "unity", "division", 
                        "polarization", "partisanship", "bipartisanship", "compromise", "consensus", "agreement", "disagreement", 
                        "opposition", "resistance", "defiance", "compliance", "conformity", "dissent", "criticism", "support", 
                        "endorsement", "approval", "disapproval", "rejection", "acceptance", "tolerance", "intolerance", 
                        "inclusion", "exclusion", "integration", "segregation", "separation", "isolation", "quarantine", 
                        "lockdown", "shutdown", "reopening", "recovery", "stimulus", "relief", "aid", "assistance", "welfare", 
                        "charity", "donation", "contribution", "tax", "tariff", "fee", "fine", "penalty", "punishment", 
                        "sentence", "imprisonment", "detention", "arrest", "charge", "accusation", "allegation", "claim", 
                        "assertion", "argument", "reasoning", "logic", "fallacy", "fact", "truth", "falsehood", "lie", 
                        "deception", "manipulation", "propaganda", "misinformation", "disinformation", "fake news", 
                        "conspiracy theory", "rumor", "gossip", "hearsay", "testimony", "witness", "evidence", "proof", 
                        "data", "statistics", "figure", "number", "percentage", "ratio", "rate", "trend", "pattern", 
                        "correlation", "causation", "effect", "impact", "influence", "consequence", "result", "outcome", 
                        "election", "campaign", "candidate", "nominee", "incumbent", "challenger", "winner", "loser", 
                        "victory", "defeat", "landslide", "narrow", "close", "contested", "disputed", "rigged", "fair", 
                        "free", "democratic", "authoritarian", "totalitarian", "dictatorial", "fascist", "communist", 
                        "socialist", "capitalist", "conservative", "liberal", "progressive", "moderate", "centrist", 
                        "radical", "extremist", "nationalist"]}

# Function to classify a title
def classify_title(title, categories):
    title_lower = title.lower()
    
    # Check for direct keyword matches
    scores = {}
    for category, keywords in categories.items():
        score = sum(1 for keyword in keywords if keyword.lower() in title_lower)
        scores[category] = score
    
    # If we have a clear match, return it
    max_score = max(scores.values())
    if max_score > 0:
        best_categories = [cat for cat, score in scores.items() if score == max_score]
        return best_categories[0]
    
    # If no direct match, return "Other"
    return "Other"

# Use SBERT for semantic understanding of titles
print("Loading SBERT model and generating embeddings...")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
title_embeddings = sbert_model.encode(cleaned_titles, convert_to_numpy=True)

# Generate category embeddings (average of multiple keywords)
category_embeddings = {}
for category, keywords in predefined_categories.items():
    keyword_embeddings = sbert_model.encode(keywords, convert_to_numpy=True)
    category_embeddings[category] = np.mean(keyword_embeddings, axis=0)

# Classify using semantic similarity as fallback for keyword matching
classifications = []
for i, title in enumerate(titles):
    # First try keyword matching
    category = classify_title(title, predefined_categories)
    
    # If no match, use semantic similarity
    if category == "Other":
        title_embedding = title_embeddings[i].reshape(1, -1)
        similarities = {}
        for cat, cat_embedding in category_embeddings.items():
            similarity = cosine_similarity(title_embedding, cat_embedding.reshape(1, -1))[0][0]
            similarities[cat] = similarity
        
        category = max(similarities.items(), key=lambda x: x[1])[0]
    
    classifications.append(category)

# Store results in a DataFrame
dataframe = pd.DataFrame({"Title": titles, "Category": classifications})

# Print category distribution
print("\n### Category Distribution ###\n")
print(dataframe["Category"].value_counts())

# Save each category separately into CSV files
for category in set(classifications):
    file_name = f"ClustersDataFrame/category_{category.replace(' & ', '_').replace(' ', '_').lower()}.csv"
    df = dataframe[dataframe["Category"] == category]
    df.to_csv(file_name, index=False)

print("\n✅ Categories saved in 'ClustersDataFrame' folder")

# Show examples from each category
print("\n### Examples from each category ###\n")
for category in set(classifications):
    print(f"--- {category} ---")
    examples = dataframe[dataframe["Category"] == category]["Title"].sample(min(5, len(dataframe[dataframe["Category"] == category]))).tolist()
    for ex in examples:
        print(f"  • {ex}")
    print()