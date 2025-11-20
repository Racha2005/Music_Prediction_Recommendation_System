import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

def build_feature_matrix(df):
    df = df.copy()

    genre_dummies = pd.get_dummies(df["Genre"])
    lang_dummies = pd.get_dummies(df["Language"])

    numeric = df[["Duration_Sec", "Popularity_Score", "Stream_Count", "Release_Year"]]

    scaler = StandardScaler()
    numeric_scaled = pd.DataFrame(
        scaler.fit_transform(numeric),
        columns=numeric.columns
    )

    features = pd.concat([genre_dummies, lang_dummies, numeric_scaled], axis=1)

    return features

def recommend(track_id, df, top_n=5):
    features = build_feature_matrix(df)

    idx = df.index[df["Track_ID"] == track_id].tolist()
    if not idx:
        raise ValueError("Track_ID not found")

    idx = idx[0]

    vec = features.iloc[idx:idx+1]

    sim = cosine_similarity(features, vec).flatten()

    df["similarity"] = sim

    recs = df.sort_values("similarity", ascending=False).iloc[1:top_n+1]

    return recs[["Track_ID", "Song_Name", "Artist", "Genre", "Language", "similarity"]]

if __name__ == "__main__":
    df = pd.read_csv("data/music_dataset.csv")
    print(recommend(10, df, top_n=5))