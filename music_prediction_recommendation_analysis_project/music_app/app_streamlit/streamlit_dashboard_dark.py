import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os

# --- CONFIG ---
st.set_page_config(layout="wide", page_title="Music Analytics — Dark Neon", page_icon="🎧")

# Local images (these were uploaded previously and are available in the runtime)
banner1 = r"/mnt/data/a86fe0fa-cbb8-46e4-9899-bfc773eb081f.png"
banner2 = r"/mnt/data/e2cac4a1-3508-48cb-bb5b-c751c6631adc.png"
banner3 = r"/mnt/data/0b630ca9-0743-4fb8-8dca-92e21fb1e59e.png"

# CSS for dark neon style and card layout
st.markdown(
    """
    <style>
    .stApp { background-color: #07090d; color: #e6f7ff; }
    .big-title { color: #00e5ff; font-size:28px; font-weight:700; margin-bottom:6px; }
    .subtle { color: #9bdcff; margin-bottom:14px; }
    .card { background: linear-gradient(90deg, rgba(18,20,30,0.85), rgba(12,14,22,0.85)); border-radius:12px; padding:12px; box-shadow: 0 6px 18px rgba(124,58,237,0.12); }
    .kpi { font-size:22px; color:#ffffff; font-weight:700; }
    .kpi-sublabel { color:#9bdcff; font-size:12px; }
    .small { font-size:12px; color:#99d7ff; }
    </style>
    """, unsafe_allow_html=True)

# Top banner images displayed across the top (white graph backgrounds inside dark container)
cols = st.columns([1,1,1])
for c,img in zip(cols, [banner1, banner2, banner3]):
    try:
        c.image(Image.open(img), use_column_width=True)
    except Exception as e:
        c.write("")

st.markdown("<div class='big-title'>Music Prediction & Recommendation Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtle'>Professional BI layout — KPIs, charts, and recommendations on a single screen</div>", unsafe_allow_html=True)

# Load dataset (relative path from project root)
data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "music_dataset_500.csv")
df = pd.read_csv(data_path)

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    gen_filter = st.multiselect("Genre", options=sorted(df['genre'].unique()), default=sorted(df['genre'].unique()))
    artist_filter = st.selectbox("Artist (optional)", options=["All"] + sorted(df['artist'].unique()))
    min_pop = st.slider("Min Popularity", 0, 100, 1)
    duration = st.slider("Duration (sec)", int(df['duration_sec'].min()), int(df['duration_sec'].max()), (120, 420))
    show_recs = st.checkbox("Show recommendations panel", value=True)
    st.markdown("---")
    st.markdown("Tip: Use filters to narrow the visuals. Charts use white backgrounds like BI reports.")

# Apply filters
q = df[df['genre'].isin(gen_filter) & (df['popularity'] >= min_pop) & (df['duration_sec'].between(duration[0], duration[1]))]
if artist_filter != "All":
    q = q[q['artist'] == artist_filter]

# TOP KPIs row
k1, k2, k3, k4, k5 = st.columns([1,1,1,1,1])
with k1:
    st.markdown("<div class='card'><div class='kpi'>{}</div><div class='kpi-sublabel'>Tracks (filtered)</div></div>".format(len(q)), unsafe_allow_html=True)
with k2:
    st.markdown("<div class='card'><div class='kpi'>{:.1f}</div><div class='kpi-sublabel'>Avg Popularity</div></div>".format(q['popularity'].mean() if len(q)>0 else 0), unsafe_allow_html=True)
with k3:
    top_gen = q['genre'].value_counts().idxmax() if len(q)>0 else "N/A"
    st.markdown("<div class='card'><div class='kpi'>{}</div><div class='kpi-sublabel'>Top Genre</div></div>".format(top_gen), unsafe_allow_html=True)
with k4:
    avg_dur = q['duration_sec'].mean() if len(q)>0 else 0
    st.markdown("<div class='card'><div class='kpi'>{:.0f}s</div><div class='kpi-sublabel'>Avg Duration</div></div>".format(avg_dur), unsafe_allow_html=True)
with k5:
    st.markdown("<div class='card'><div class='kpi'>{}</div><div class='kpi-sublabel'>Unique Artists</div></div>".format(q['artist'].nunique()), unsafe_allow_html=True)

st.markdown("---")

# Main layout: left filter summary + central charts + right recommendations
left, center, right = st.columns([0.9, 2, 1.1])

with left:
    st.markdown("<div class='card'><h4 style='color:#c3f5ff'>Filter Summary</h4>", unsafe_allow_html=True)
    st.write("Genres:", ", ".join(gen_filter))
    st.write("Artist:", artist_filter)
    st.write("Popularity ≥", min_pop)
    st.write("Duration:", f"{duration[0]} - {duration[1]} sec")
    st.markdown("</div>", unsafe_allow_html=True)

with center:
    # Big bar chart (genre distribution) with white background like BI
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    cnt = q['genre'].value_counts().reset_index()
    cnt.columns = ['genre','count']
    fig = px.bar(cnt, x='genre', y='count', text='count', title="Genre Distribution (filtered)", template='plotly_white')
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='rgba(0,0,0,0)', font_color="#0b2130")
    fig.update_traces(marker=dict(color=['#7c3aed','#00e5ff','#ff7ab6','#ffd166','#7ee081','#ffa156','#6b7cff']))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # Line chart (popularity trend by tempo buckets) — white background
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    df_temp = q.copy()
    df_temp['tempo_bucket'] = pd.cut(df_temp['tempo'], bins=6)
    line = df_temp.groupby('tempo_bucket')['popularity'].mean().reset_index()
    line['tempo_label'] = line['tempo_bucket'].astype(str)
    fig2 = px.line(line, x='tempo_label', y='popularity', markers=True, title="Avg Popularity by Tempo Bucket", template='plotly_white')
    fig2.update_layout(plot_bgcolor='white', paper_bgcolor='rgba(0,0,0,0)', font_color="#0b2130")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Top 10 Tracks")
    top = q.sort_values("popularity", ascending=False).head(10)
    st.table(top[['track_id','title','artist','genre','popularity']].reset_index(drop=True))
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# Bottom row: Donut charts + histogram + recommendations table (white chart backgrounds)
d1, d2, d3 = st.columns([1,1,1])
with d1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Genre Share")
    pie = q['genre'].value_counts().reset_index()
    pie.columns = ['genre','count']
    figp = px.pie(pie, values='count', names='genre', hole=0.5, title="", template='plotly_white')
    figp.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(figp, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with d2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Mood Share (synthetic)")
    qq = q.copy()
    qq['mood'] = ((qq['valence']>0.5) & (qq['energy']>0.5)).map({True:'Positive', False:'Neutral/Calm'})
    mood = qq['mood'].value_counts().reset_index()
    mood.columns = ['mood','count']
    figm = px.pie(mood, values='count', names='mood', hole=0.4, template='plotly_white')
    st.plotly_chart(figm, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with d3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Energy Distribution")
    figh = px.histogram(q, x='energy', nbins=20, title="", template='plotly_white')
    figh.update_layout(plot_bgcolor='white', paper_bgcolor='rgba(0,0,0,0)', font_color="#0b2130")
    st.plotly_chart(figh, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# Recommendations panel with export option
if show_recs:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Recommendations (content-based)")
    seed_track = st.selectbox("Seed track", options=df['track_id'].tolist(), index=0)
    seed = df[df['track_id']==seed_track].iloc[0]
    def score_row(r):
        s = 0
        s += 2.0 if r['genre'] == seed['genre'] else 0.0
        s -= abs(r['popularity'] - seed['popularity'])/100.0
        s -= sum(abs(r[f]-seed[f]) for f in ['danceability','energy','acousticness','valence'])
        return s
    df['score_tmp'] = df.apply(score_row, axis=1)
    recs = df.sort_values('score_tmp', ascending=False).query("track_id!=@seed_track").head(12)
    st.dataframe(recs[['track_id','title','artist','genre','popularity']])
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='color:#808f9a; font-size:12px'>Note: Charts use white backgrounds inside cards to mimic BI report visuals.</div>", unsafe_allow_html=True)
