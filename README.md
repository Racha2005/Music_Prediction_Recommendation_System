# 🎵 Music Prediction & Recommendation Analysis Project (Dark Neon)

### ▶ Run the Streamlit Dashboard
```
streamlit run music_app/app_streamlit/streamlit_dashboard_dark.py
```
### ▶ Run Desktop Demo
```
python -m music_app.app_desktop.desktop_app
```
# or  
```
python music_app/app_desktop/desktop_app.py
```

### ▶ Train ML Models
```
python -m music_app.ml.train_popularity  
python -m music_app.ml.train_mood
```


# 🎧 MUSIC PREDICTION AND RECOMMENDATION SYSTEM

A complete end-to-end **Music Intelligence System** featuring:

- 🌐 Streamlit Analytics Dashboard  
- 💻 Tkinter Desktop App  
- 🤖 ML Models (Popularity + Mood)  
- 🎧 Audio Feature Extraction  
- 🖼️ Album Cover Image Hashing  
- 📝 Lyrics Sentiment Analysis  
- 🔍 UMAP + KMeans Clustering  
- 🎼 Content-Based Recommendation Engine  
- 🌙 Dark Neon Theme  

Transforms raw music data into insights, predictions, and recommendations.



## ⭐ OVERVIEW

This system helps users:
- Analyze music trends  
- Predict track popularity  
- Understand mood & energy  
- Explore genre distributions  
- Use BI-style dashboards  
- Get smart recommendations  
- Analyze audio, images, and lyrics  
- Use both web and desktop interfaces  



## 📁 FOLDER STRUCTURE
```
music_prediction_recommendation_analysis_project/  
│  
├── setup.py  
├── pyproject.toml  
├── requirements.txt  
│  
├── data/  
│   ├── music_dataset_500.csv  
│   ├── audio_features.csv  
│   ├── image_hashes.csv  
│   ├── lyrics_analysis.csv  
│   ├── clusters.csv  
│   ├── covers/  
│   └── audio/  
│  
└── music_app/  
    ├── config.py  
    ├── cli.py  
    │  
    ├── models/  
    │   ├── popularity_rf.joblib  
    │   └── mood_rf.joblib  
    │  
    ├── ml/  
    │   ├── train_popularity.py  
    │   ├── train_mood.py  
    │   └── recommender.py  
    │  
    ├── extras/  
    │   ├── audio_features.py  
    │   ├── image_hashing.py  
    │   ├── lyrics_analysis.py  
    │   ├── clustering.py  
    │   ├── recommenders.py  
    │   ├── organize_music.py  
    │   └── voice_commands.py  
    │  
    ├── app_streamlit/  
    │   └── streamlit_dashboard_dark.py  
    │  
    ├── app_desktop/  
    │   └── desktop_app.py  
    │  
    └── app_flask/  
        ├── app.py  
        └── templates/  
            └── index.html  
```


## 🚀 RUN COMMANDS

### Activate Virtual Environment
```
.\venv\Scripts\Activate.ps1
```

### Run Streamlit Dashboard
```
streamlit run music_app/app_streamlit/streamlit_dashboard_dark.py
```

### Run Desktop Application
```
python music_app/app_desktop/desktop_app.py
```

### Train Models
```
python -m music_app.ml.train_popularity  
python -m music_app.ml.train_mood
```

### Extract Audio Features
```
python music_app/extras/audio_features.py --folder data/audio --out data/audio_features.csv
```

### Generate Image Hashes
```
python music_app/extras/image_hashing.py --folder data/covers --out data/image_hashes.csv
```

### Lyrics Sentiment Analysis
```
python music_app/extras/lyrics_analysis.py --csv data/music_dataset_500.csv --out data/lyrics_analysis.csv
```

### UMAP + KMeans Clustering
```
python music_app/extras/clustering.py --features data/audio_features.csv --out data/clusters.csv --n 8
```


## 📊 FEATURES

### 1. Streamlit Dashboard (Dark Neon)
- KPI Cards  
- Genre distribution (bar + donut)  
- Popularity vs tempo trends  
- Mood pie chart  
- Energy histogram  
- Filters (genre, artist, popularity, duration)  
- Recommendations table  
- Image hashing (grid view + similar artwork viewer)  
- Audio features table  
- Lyrics sentiment table  
- Song clustering visualization  

### 2. Tkinter Desktop App
- Top tracks list  
- Genre distribution chart  
- Recommendations popup  
- Audio import option  



## 🤖 MACHINE LEARNING MODELS

### Popularity Model
- RandomForestRegressor  
- Inputs: energy, valence, acousticness, danceability, tempo, duration  

### Mood Model
- RandomForestClassifier  
- Output: Positive / Neutral-Calm  

### Content-Based Recommender
Similarity Score =  
Genre Match − Popularity Difference − Feature Distance  



## 📦 TECHNOLOGIES USED

### Core
Python, Streamlit, Tkinter  

### Data & ML
pandas, numpy, scikit-learn, joblib  

### Audio / Text / Image
librosa, nltk, pillow, imagehash  

### Clustering
UMAP, KMeans  



## 🎼 DATASET INFO

Contains **500 tracks** with:  
- Genre, Artist, Title, Duration, Tempo  
- Popularity score  
- Energy, Valence, Danceability  
- Acousticness, Loudness  

Perfect for ML + visualization.



## 🎯 PURPOSE

Ideal for:
- Internships  
- College major projects  
- Portfolio-ready ML systems  
- Learning ML + Visualization + GUI  



## 🔥 KEY OUTCOMES

- BI-style analytics dashboard  
- Cross-platform desktop application  
- ML predictions (popularity + mood)  
- Audio, image, lyrics analysis  
- Complete end-to-end music intelligence workflow  
## 📄 LICENSE

MIT License
