import streamlit as st
import pandas as pd
import joblib
from music_ml.recommender import recommend

st.title("🎵 Music Analytics & Recommendation System")

df = pd.read_csv("data/music_dataset.csv")

menu = st.sidebar.selectbox("Choose Option", [
    "Explore Dataset",
    "Popularity Prediction",
    "Song Recommender",
    "Mood Classification"
])

if menu == "Explore Dataset":
    st.dataframe(df.head(100))
    st.bar_chart(df["Genre"].value_counts())

elif menu == "Popularity Prediction":
    model_data = joblib.load("models/music_pop_model.pkl")
    model = model_data["model"]
    enc = model_data["encoders"]

    st.subheader("Predict Popularity")
    genre = st.selectbox("Genre", df["Genre"].unique())
    lang = st.selectbox("Language", df["Language"].unique())
    artist = st.selectbox("Artist", df["Artist"].unique())

    duration = st.slider("Duration (sec)", 90, 600)
    year = st.number_input("Release Year", 1990, 2025)
    streams = st.number_input("Stream Count", 0, 10000000)

    if st.button("Predict"):
        g = enc["genre"].transform([genre])[0]
        l = enc["lang"].transform([lang])[0]
        a = enc["artist"].transform([artist])[0]

        X = [[a, g, l, duration, year, streams]]
        pred = model.predict(X)[0]
        st.success(f"Predicted Popularity Score: {pred:.2f}")

elif menu == "Song Recommender":
    st.subheader("Find Similar Songs")
    track_id = st.number_input("Track ID", 1, len(df), 10)
    topn = st.slider("Top N", 1, 20, 5)
    recs = recommend(track_id, df, top_n=topn)
    st.write(recs)

elif menu == "Mood Classification":
    st.subheader("Mood Classifier")
    st.write("Run: python music_ml/train_mood.py to generate mood_clf.pkl")