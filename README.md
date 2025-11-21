# **Music Prediction & Recommendation Analysis**

This project is a complete, end-to-end music intelligence system combining data analytics, professional data visualization, machine-learning predictions, and an interactive recommendation engine.  
It includes two interfaces — a **Streamlit Dashboard** and a **Tkinter Desktop Application** — both styled with a modern **Dark Neon** theme to provide a clean and engaging visual experience.

The goal of this project is to demonstrate how raw music data can be transformed into meaningful insights and user-facing tools using data science.



## **Overview**

This project focuses on understanding the hidden patterns within music datasets and creating tools that help users:

- Visualize and explore music characteristics  
- Understand trends in genres, mood, energy, and popularity  
- Predict important attributes such as track popularity  
- Receive song recommendations based on similarity  
- Use analytics in both web and desktop environments  

The entire system is built to be simple to use, while still being technically rich, making it ideal for students, researchers, and data science enthusiasts.



## **What This Project Includes**

### **1. Streamlit Dashboard (Dark-Neon BI Theme)**  
The Streamlit dashboard acts as a complete analytics and recommendation platform. It contains:

#### **Interactive KPI Cards**
Summarizes essential insights like:
- Total tracks  
- Average popularity  
- Top genre  
- Average duration  
- Number of artists  

Provides a quick snapshot of the dataset.

#### **Genre Distribution Analysis**
Bar charts and donut charts show:
- Which genres dominate the dataset  
- How users can explore filtered genre counts  

#### **Popularity & Mood Trends**
Line charts and category breakdowns display:
- How popularity varies with tempo  
- Mood distribution based on energy + valence  

Helps users understand overall listening patterns.

#### **Energy & Acoustic Insights**
Visual analysis showing:
- Energy levels across tracks  
- Acousticness patterns  
- Listener engagement possibilities  

#### **White-Background BI Charts**
Charts follow a modern BI layout:
- Charts inside white sections  
- Surrounded by a dark background  
- Creates a professional dashboard style  

#### **Explorer Filters**
Users can refine the dataset by:
- Genre  
- Artist  
- Popularity level  
- Track duration  

These filters help create custom visual insights.

#### **Recommendation Panel**
A data-driven content-based recommender suggesting music similar to a selected track.  
Similarity is based on:
- Genre  
- Energy  
- Danceability  
- Acousticness  
- Valence  
- Popularity difference  



### **2. Tkinter Desktop Application**

The desktop app allows users to experience a simplified version of the project, ideal for offline use.

#### **Top Tracks Viewer**
Displays the most popular tracks with details such as:
- Track ID  
- Title  
- Genre  
- Popularity score  

Helps users quickly browse top-performing songs.

#### **Genre Distribution Chart**
A professional-style bar chart embedded into the Tkinter window using Matplotlib.  
Shows the distribution of tracks across different genres.

#### **Recommendation Popup Window**
When a track is selected, the system:
1. Evaluates similarity based on numerical features  
2. Displays top recommended tracks in a popup  
3. Shows title, genre, and popularity score  

The desktop interface is intentionally lightweight and user-friendly.

#### **Dark Neon Aesthetic**
Both UI frames and components use:
- Dark backgrounds  
- Cyan text  
- Modern dashboard-style styling  



### **3. Machine Learning Components**

This project includes machine-learning scripts integrated to enhance functionality.

#### **Popularity Prediction**
Model: RandomForestRegressor  
Purpose:
- Predict whether a song is likely to be popular  
- Helps in analytics or playlist ranking  

#### **Mood Classification**
Tracks are classified into mood categories using:
- Energy  
- Valence  
- Acousticness  
- Danceability  

Mood categories like:
- Positive  
- Calm / Neutral  
Helps generate playlist suggestions.

#### **Content-Based Recommender**
A key feature of the project.  
Compares songs using:
- Genre match (priority weighted)  
- Popularity proximity  
- Feature similarity:
  - Energy  
  - Valence  
  - Danceability  
  - Acousticness  

Outputs a ranked list of most similar tracks.



## **Packages Used**

### **Core Libraries**
- **pandas**  
  - Data cleaning, filtering, transformation  
- **numpy**  
  - Statistical computation  
- **matplotlib**  
  - Charting system for desktop app  
- **plotly**  
  - Interactive charts for Streamlit  
- **streamlit**  
  - Builds the dashboard interface  
- **tkinter**  
  - Creates desktop UI  

### **Machine Learning**
- **scikit-learn**  
  - Random Forest models  
  - Train-test splitting  
- **joblib**  
  - Saving/loading ML models  

### **Supporting Tools**
- **pathlib** – flexible file path handling  
- **os** – system-level file utilities  
- **Pillow (PIL)** – loading banner images  
- **seaborn** *(optional)* – advanced visuals  

These combined tools allow the project to seamlessly integrate analytics, ML, and UI.



## **Dataset Details**

The dataset contains **500 tracks** with detailed features:

### **Acoustic Features**
- Acousticness  
- Energy  
- Danceability  
- Valence  
- Tempo  
- Loudness  
- Instrumentalness  

### **Categorical/Music Metadata**
- Genre  
- Artist  
- Track ID  
- Song title  

### **Popularity Metrics**
- Popularity score (0–100)

The dataset is clean, consistent, and suitable for:
- Exploratory data analysis  
- ML modeling  
- Interactive dashboards  



## **Purpose of the Project**

This project demonstrates a full real-world workflow:

1. **Data Collection**  
2. **Cleaning & Preprocessing**  
3. **Visualization & Analytics**  
4. **Machine Learning Modeling**  
5. **Recommendation Engine**  
6. **Multi-interface Deployment** (Web + Desktop)

It is especially valuable for:
- Portfolio showcases  
- Academic submissions  
- Internship technical projects  
- Learning ML + visualization technologies  



## **Additional Information**

### **Dark Neon Theme**
Chosen because it:
- Enhances contrast  
- Gives a professional BI appearance  
- Improves readibility of charts  
- Matches modern dashboard design trends  

### **Why Streamlit + Tkinter Together?**
- Streamlit → deep, interactive analytics  
- Tkinter → fast offline tool  

Combining both demonstrates versatility.

### **Recommendation System Logic**
The similarity score =  
**Genre Match (weight)  
− Popularity Difference  
− Sum of Feature Distances**  

Which makes it:
- Simple  
- Fast  
- Effective  

### **Machine Learning Benefits**
- Predicts potential hit songs  
- Groups songs by mood clusters  
- Provides quantitative support for recommendations  



## **Key Outcomes**

- Professional Dark-Neon music analytics dashboard  
- Fully functional Tkinter desktop app  
- Trained ML models (Popularity + Mood)  
- Smooth recommendation engine  
- Easy-to-understand visuals  
- Clean, modern system design  
- Shows complete end-to-end project execution  



## **Summary**

This project provides a complete music analysis ecosystem.  
Users can explore track trends, visualize characteristics, predict popularity, analyze mood and energy, and get intelligent recommendations.  
With its Dark Neon theme, machine learning integration, and dual-interface design, the system is both **practical and presentation-ready**, suitable for both academic and professional use.

