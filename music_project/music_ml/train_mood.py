import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

df = pd.read_csv("data/music_dataset.csv")

le_genre = LabelEncoder().fit(df["Genre"])
le_lang = LabelEncoder().fit(df["Language"])
le_artist = LabelEncoder().fit(df["Artist"])
le_mood = LabelEncoder().fit(df["Mood"])

X = pd.DataFrame({
    "genre": le_genre.transform(df["Genre"]),
    "lang": le_lang.transform(df["Language"]),
    "artist": le_artist.transform(df["Artist"]),
    "duration": df["Duration_Sec"],
    "year": df["Release_Year"],
    "streams": df["Stream_Count"],
})

y = le_mood.transform(df["Mood"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=250, random_state=42)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
print(classification_report(y_test, pred, target_names=le_mood.classes_))

os.makedirs("models", exist_ok=True)

joblib.dump({
    "clf": clf,
    "encoders": {
        "genre": le_genre,
        "lang": le_lang,
        "artist": le_artist,
        "mood": le_mood
    }
}, "models/mood_clf.pkl")

print("Mood classifier saved.")