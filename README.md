# Music Prediction & Recommendation Analysis Project (Dark Neon)

Run the Streamlit dashboard: `streamlit run music_app/app_streamlit/streamlit_dashboard_dark.py` from the project root.

Run desktop demo: `python -m music_app.app_desktop.desktop_app` (or `python music_app/app_desktop/desktop_app.py`)

Train models: python -m music_app.ml.train_popularity` and `python -m music_app.ml.train_mood
====

# **"Music Prediction and Recommendation System"**

This project is a complete, end-to-end music intelligence system combining data analytics, professional data visualization, machine-learning predictions, and an interactive recommendation engine.  
It includes two interfaces — a **Streamlit Dashboard** and a **Tkinter Desktop Application** — both styled with a modern theme to provide a clean and engaging visual experience.

The goal of this project is to demonstrate how raw music data can be transformed into meaningful insights and user-facing tools using data science.



## **Overview**

This project focuses on understanding the hidden patterns within music datasets and creating tools that help users:

- Visualize and explore music characteristics  
- Understand trends in genres, mood, energy, and popularity  
- Predict important attributes such as track popularity  
- Receive song recommendations based on similarity  
- Uses data analytics in both web and desktop environments  

The entire system is built to be simple to use, while still being technically rich, making it ideal for students, researchers, and data science enthusiasts.



## **Folder Structure**

```
music_prediction_recommendation_analysis_project/
│
├── setup.py
├── pyproject.toml
├── requirements.txt
│
├── data/
│   └── music_dataset_500.csv
│
└── music_app/
    ├── config.py
    ├── cli.py
    │
    ├── ml/
    │   ├── train_popularity.py
    │   ├── train_mood.py
    │   ├── recommender.py
    │
    ├── data_analysis/
    │   ├── eda_report.py
    │   └── trends.py
    │
    ├── app_streamlit/
    │   ├── streamlit_dashboard_dark.py
    │   └── streamlit_app.py
    │
    ├── app_desktop/
    │   └── desktop_app.py
    │
    └── app_flask/
        ├── app.py
        └── templates/
            └── index.html
```



## **Run Commands**

### **Activate Virtual Environment**
```
.\.venv\Scripts\Activate.ps1
```

### **Run Streamlit Dashboard**
```
streamlit run music_app/app_streamlit/streamlit_dashboard_dark.py
```

### **Run Desktop Application**
```
python music_app/app_desktop/desktop_app.py
```

### **Train Popularity Model**
```
python -m music_app.ml.train_popularity
```

### **Train Mood Classification Model**
```
python -m music_app.ml.train_mood
```

---

## **What This Project Includes**

### **1. Streamlit Dashboard**
The Streamlit dashboard acts as a complete analytics and recommendation platform. It contains:

#### **Interactive KPI Cards**
Summarizes essential insights like:
- Total tracks  
- Average popularity  
- Top genre  
- Average duration  
- Number of artists  

#### **Genre Distribution Analysis**
- Which genres dominate the dataset  
- Filterable visualizations  
- Donut and bar charts  

#### **Popularity & Mood Trends**
Shows:
- Popularity variations  
- Mood distribution by energy + valence  

#### **Energy & Acoustic Insights**
Displays:
- Energy levels  
- Acousticness patterns  

#### **BI-Style White-Background Charts**
Charts placed inside white containers for professional dashboard styling.

#### **Explorer Filters**
Users can filter by:
- Genre  
- Artist  
- Popularity level  
- Track duration  

#### **Recommendation Panel**
The content-based recommender suggests similar tracks using:
- Genre  
- Energy  
- Danceability  
- Acousticness  
- Valence  
- Popularity difference  



### **2. Tkinter Desktop Application**

#### **Top Tracks Viewer**
Displays:
- Track ID  
- Title  
- Genre  
- Popularity  

#### **Genre Distribution Chart**
A Matplotlib-based chart embedded inside Tkinter.

#### **Recommendation Popup Window**
Displays similar tracks by:
- Title  
- Genre  
- Popularity score  

#### **UI Aesthetic**
- Dark backgrounds  
- Cyan text  
- Clean layout  



### **3. Machine Learning Components**

#### **Popularity Prediction**
- Model: RandomForestRegressor  
- Predicts future popularity  
- Useful for understanding trending potential  

#### **Mood Classification**
Uses:
- Energy  
- Valence  
- Acousticness  
- Danceability  

Outputs:
- Positive  
- Calm / Neutral  

#### **Content-Based Recommender**
Scores similarity using:
- Genre match  
- Popularity difference  
- Feature similarity  
  - Energy  
  - Valence  
  - Danceability  
  - Acousticness  



## **Packages Used**

### **Core Libraries**
- pandas  
- numpy  
- matplotlib  
- plotly  
- streamlit  
- tkinter  

### **Machine Learning**
- scikit-learn  
- joblib  

### **Supporting Tools**
- pathlib  
- os  
- Pillow (PIL)  
- seaborn *(optional)*  

These enable integrated analytics, ML, visualization, and UI workflows.



## **Dataset Details**

The dataset contains **500 tracks** with:

### **Acoustic Features**
- Acousticness  
- Energy  
- Danceability  
- Valence  
- Tempo  
- Loudness  
- Instrumentalness  

### **Metadata**
- Genre  
- Artist  
- Track ID  
- Song title  

### **Popularity Metrics**
- Popularity score (0–100)

The dataset is clean, structured, and ideal for analytics and ML.



## **Purpose of the Project**

This project demonstrates a full real-world workflow:

1. **Data Collection**  
2. **Cleaning & Preprocessing**  
3. **Visualization**  
4. **Machine Learning**  
5. **Recommendation Engine**  
6. **Dashboard + Desktop Deployment**  

Perfect for:
- Academic submissions  
- Portfolio projects  
- Internship technical tasks  
- Learning ML + visualization  



## **Additional Information**

### **Streamlit + Tkinter Combination**
- Streamlit → deep, interactive analytics  
- Tkinter → quick, offline-friendly UI  

### **Recommendation System Logic**
Similarity Score:
```
Genre Match Weight 
− Popularity Difference 
− Feature Distance (Energy, Valence, Danceability, Acousticness)
```

## **Key Outcomes**

- Fully functional music analytics dashboard  
- Tkinter desktop tool for offline usage  
- Trained ML models  
- Data-driven recommendations  
- Modern UI design  
- Easy-to-understand visual reports  
- End-to-end practical implementation  



## **Summary**

This project provides a complete music analysis ecosystem.  
Users can explore track trends, visualize characteristics, predict popularity, analyze mood and energy, and receive intelligent recommendations.  
It is professional, modern, interactive, and perfect for academic or portfolio use.
