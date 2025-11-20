import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/music_dataset.csv")

# -------------------------
# GRAPH 1 – Genre Popularity Over Time
# -------------------------
genre_year = df.groupby(["Release_Year", "Genre"])["Popularity_Score"].mean().unstack().fillna(0)
plt.figure(figsize=(14,7))
genre_year.plot()
plt.title("Genre Popularity Over Years")
plt.xlabel("Year")
plt.ylabel("Average Popularity")
plt.tight_layout()
plt.show()

# -------------------------
# GRAPH 2 – Top 15 Artists by Average Popularity
# -------------------------
artist_pop = df.groupby("Artist")["Popularity_Score"].mean().sort_values(ascending=False).head(15)
plt.figure(figsize=(12,6))
sns.barplot(x=artist_pop.values, y=artist_pop.index)
plt.title("Top 15 Artists by Popularity")
plt.xlabel("Average Popularity")
plt.tight_layout()
plt.show()

# -------------------------
# GRAPH 3 – Popularity Score Distribution
# -------------------------
plt.figure(figsize=(10,5))
sns.histplot(df["Popularity_Score"], bins=20, kde=True)
plt.title("Distribution of Popularity Scores")
plt.xlabel("Popularity")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# -------------------------
# GRAPH 4 – Streams vs Popularity
# -------------------------
plt.figure(figsize=(10,6))
sns.scatterplot(x=df["Popularity_Score"], y=df["Stream_Count"])
plt.title("Streams vs Popularity")
plt.xlabel("Popularity Score")
plt.ylabel("Stream Count")
plt.tight_layout()
plt.show()

# -------------------------
# GRAPH 5 – Mood Distribution
# -------------------------
mood_counts = df["Mood"].value_counts()
plt.figure(figsize=(8,6))
plt.pie(mood_counts.values, labels=mood_counts.index, autopct="%1.1f%%")
plt.title("Mood Distribution")
plt.tight_layout()
plt.show()