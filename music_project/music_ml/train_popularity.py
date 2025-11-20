import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import os

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    df = df.copy()

    le_artist = LabelEncoder()
    le_genre = LabelEncoder()
    le_lang = LabelEncoder()

    df["artist_enc"] = le_artist.fit_transform(df["Artist"])
    df["genre_enc"] = le_genre.fit_transform(df["Genre"])
    df["lang_enc"] = le_lang.fit_transform(df["Language"])

    X = df[[
        "artist_enc", "genre_enc", "lang_enc",
        "Duration_Sec", "Release_Year", "Stream_Count"
    ]]
    y = df["Popularity_Score"]

    encoders = {
        "artist": le_artist,
        "genre": le_genre,
        "lang": le_lang
    }

    return X, y, encoders

def train_and_save(model_out="models/music_pop_model.pkl"):
    df = load_data("data/music_dataset.csv")

    X, y, encoders = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=250, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("MAE:", mean_absolute_error(y_test, preds))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))
    print("R2 Score:", r2_score(y_test, preds))

    os.makedirs("models", exist_ok=True)

    joblib.dump({
        "model": model,
        "encoders": encoders
    }, model_out)

    print("Model saved to", model_out)

if __name__ == "__main__":
    train_and_save()