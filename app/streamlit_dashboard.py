import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Netflix Data Analysis & Insights",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CINEMATIC LANDING INTRO ---
st.markdown("""
<style>
/* Cinematic Overlay Container */
#cinematic-intro {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: #050505;
  z-index: 9999999;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  animation: fadeOutIntro 8s forwards;
}

@keyframes fadeOutIntro {
  0% { opacity: 1; visibility: visible; }
  85% { opacity: 1; visibility: visible; }
  100% { opacity: 0; visibility: hidden; pointer-events: none; }
}

/* Netflix 'N' Logo */
.netflix-n-container {
  position: relative;
  width: 80px;
  height: 130px;
  transform: scale(0);
  animation: scaleInN 1s cubic-bezier(0.19, 1, 0.22, 1) forwards 0.5s, zoomOutN 1.5s ease-in-out forwards 4.5s;
}

@keyframes scaleInN {
  0% { transform: scale(0); opacity: 0; }
  100% { transform: scale(1); opacity: 1; }
}

@keyframes zoomOutN {
  0% { transform: scale(1); opacity: 1; filter: blur(0px); }
  100% { transform: scale(4); opacity: 0; filter: blur(20px); }
}

.netflix-n {
  position: relative;
  width: 100%;
  height: 100%;
}

.netflix-n div {
  position: absolute;
  top: 0;
  height: 100%;
  background: #E50914;
  border-radius: 2px;
  box-shadow: 0 0 15px rgba(229, 9, 20, 0.4);
}

.n-left { left: 0; width: 26px; z-index: 2; }
.n-right { right: 0; width: 26px; z-index: 1; }
.n-center {
  left: 9px;
  width: 30px;
  height: 140px;
  transform: rotate(-24deg);
  transform-origin: top left;
  z-index: 3;
  box-shadow: 0 0 20px rgba(0,0,0,0.8);
  background: linear-gradient(to right, #E50914 0%, #ff4b55 50%, #B81D24 100%);
}

/* Portfolio Text Reveal */
.intro-text {
  position: absolute;
  bottom: 25%;
  text-align: center;
  color: white;
  opacity: 0;
  animation: fadeInText 1.5s forwards 2.5s, fadeOutText 1.5s forwards 5.5s;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

@keyframes fadeInText {
  0% { opacity: 0; transform: translateY(30px); }
  100% { opacity: 1; transform: translateY(0); }
}

@keyframes fadeOutText {
  0% { opacity: 1; transform: translateY(0); filter: blur(0px); }
  100% { opacity: 0; transform: translateY(-30px); filter: blur(10px); }
}

.intro-name {
  font-size: 2.5rem;
  letter-spacing: 0.3em;
  margin-bottom: 8px;
  font-weight: 800;
  text-shadow: 0 0 15px rgba(255,255,255,0.2);
}

.intro-title {
  font-size: 0.9rem;
  color: #aaa;
  letter-spacing: 0.15em;
  margin-bottom: 25px;
  font-weight: 500;
}

.intro-subtitle {
  font-size: 1.5rem;
  font-weight: 700;
  background: -webkit-linear-gradient(45deg, #E50914, #ff4b55);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
</style>

<div id="cinematic-intro">
  <div class="netflix-n-container">
    <div class="netflix-n">
      <div class="n-left"></div>
      <div class="n-center"></div>
      <div class="n-right"></div>
    </div>
  </div>
  <div class="intro-text">
    <div class="intro-name">SHRUTI SINGH</div>
    <div class="intro-title">DATA SCIENTIST • AI DEVELOPER</div>
    <div class="intro-subtitle">Transforming Data into Intelligent Decisions</div>
  </div>
</div>
""", unsafe_allow_html=True)

# --- CUSTOM CSS (NETFLIX THEME & GLASSMORPHISM) ---
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background-color: #141414;
        color: #FFFFFF;
    }
    
    /* Hide Streamlit Header/Footer */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Netflix Red Highlight */
    .st-bb {
        background-color: #E50914;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #000000;
        border-right: 1px solid #333;
    }
    
    /* KPI Cards Glassmorphism */
    .kpi-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        transition: transform 0.3s;
    }
    .kpi-card:hover {
        transform: translateY(-5px);
    }
    .kpi-title {
        color: #aaa;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .kpi-value {
        color: #E50914;
        font-size: 36px;
        font-weight: bold;
        margin-top: 10px;
    }
    
    /* Section Headers */
    .section-header {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #FFFFFF;
        font-weight: 700;
        border-left: 5px solid #E50914;
        padding-left: 10px;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    
    /* Business Insight Box */
    .insight-box {
        background: rgba(229, 9, 20, 0.1);
        border-left: 4px solid #E50914;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        font-size: 15px;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADING & CLEANING ---
@st.cache_data
def load_and_clean_data():
    data_path = os.path.join(os.path.dirname(__file__), '../data/netflix_titles.csv')
    df = pd.read_csv(data_path)
    
    # 1. Handle Missing Values
    df['director'] = df['director'].fillna('Unknown')
    df['cast'] = df['cast'].fillna('Unknown')
    df['country'] = df['country'].fillna('Unknown')
    df['date_added'] = df['date_added'].fillna(df['date_added'].mode()[0])
    df['rating'] = df['rating'].fillna(df['rating'].mode()[0])
    df['duration'] = df['duration'].fillna('0')
    
    # 2. Date Conversions
    df['date_added'] = pd.to_datetime(df['date_added'].str.strip(), format='%B %d, %Y', errors='coerce')
    df['year_added'] = df['date_added'].dt.year.fillna(2020).astype(int)
    df['month_added'] = df['date_added'].dt.month_name()
    
    # 3. Standardize Country (Take first country if multiple)
    df['primary_country'] = df['country'].apply(lambda x: x.split(',')[0].strip())
    
    # 4. Extract Duration Numeric
    df['duration_num'] = df['duration'].str.extract('(\d+)').astype(float)
    
    # 5. ML Feature Engineering (Combined text for recommendation)
    df['combined_features'] = df['title'] + ' ' + df['director'] + ' ' + df['cast'] + ' ' + df['listed_in'] + ' ' + df['description']
    
    return df

try:
    df = load_and_clean_data()
except Exception as e:
    st.error(f"Error loading data: {e}. Please ensure the dataset is downloaded in the data/ folder.")
    st.stop()

# --- SIDEBAR FILTERS ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", width=150)
st.sidebar.markdown("---")
st.sidebar.header("Filter Content")

type_filter = st.sidebar.multiselect("Content Type", options=df['type'].unique(), default=df['type'].unique())
year_range = st.sidebar.slider("Release Year", min_value=int(df['release_year'].min()), max_value=int(df['release_year'].max()), value=(2000, 2021))
top_countries = df['primary_country'].value_counts().head(20).index.tolist()
country_filter = st.sidebar.multiselect("Country", options=top_countries, default=[])

# Apply filters
filtered_df = df[(df['type'].isin(type_filter)) & 
                 (df['release_year'] >= year_range[0]) & 
                 (df['release_year'] <= year_range[1])]

if country_filter:
    filtered_df = filtered_df[filtered_df['primary_country'].isin(country_filter)]

# --- MAIN DASHBOARD ---
st.markdown("<h1 style='text-align: center; color: #E50914;'>NETFLIX EXECUTIVE DASHBOARD</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #aaa; margin-bottom: 40px;'>Comprehensive Analysis & Intelligence Platform</p>", unsafe_allow_html=True)

# KPI Section
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Total Titles</div><div class='kpi-value'>{len(filtered_df):,}</div></div>", unsafe_allow_html=True)
with col2:
    movies = len(filtered_df[filtered_df['type'] == 'Movie'])
    st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Movies</div><div class='kpi-value'>{movies:,}</div></div>", unsafe_allow_html=True)
with col3:
    shows = len(filtered_df[filtered_df['type'] == 'TV Show'])
    st.markdown(f"<div class='kpi-card'><div class='kpi-title'>TV Shows</div><div class='kpi-value'>{shows:,}</div></div>", unsafe_allow_html=True)
with col4:
    countries_count = filtered_df['primary_country'].nunique()
    st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Global Reach (Countries)</div><div class='kpi-value'>{countries_count}</div></div>", unsafe_allow_html=True)

st.markdown("---")

# Tabs for different analysis sections
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview & Trends", "Geographic Intelligence", "Genre & Content", "Cast & Directors", "AI Recommendations"])

# --- TAB 1: OVERVIEW & TRENDS ---
with tab1:
    st.markdown("<div class='section-header'>Platform Growth & Release Trends</div>", unsafe_allow_html=True)
    
    colA, colB = st.columns([2, 1])
    
    with colA:
        # Content Added Per Year
        yearly_added = filtered_df.groupby(['year_added', 'type']).size().reset_index(name='count')
        fig_yearly = px.area(yearly_added, x='year_added', y='count', color='type', 
                             title="Content Added Over Time",
                             color_discrete_map={'Movie': '#E50914', 'TV Show': '#564d4d'},
                             template='plotly_dark')
        fig_yearly.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_yearly, use_container_width=True)
        
    with colB:
        # Movies vs TV Shows Distribution
        fig_pie = px.pie(filtered_df, names='type', title="Content Distribution",
                         color='type', color_discrete_map={'Movie': '#E50914', 'TV Show': '#221f1f'},
                         hole=0.5, template='plotly_dark')
        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pie, use_container_width=True)
        
    st.markdown("""
    <div class='insight-box'>
        <strong>Executive Insight:</strong> The platform experienced exponential growth in content acquisition starting in 2015, peaking around 2019. While Movies dominate the overall catalog (~70%), TV Shows have seen a more aggressive growth rate in recent years as Netflix shifts focus to original series retention.
    </div>
    """, unsafe_allow_html=True)

    # Content release by month heatmap
    st.markdown("<div class='section-header'>Seasonality Analysis</div>", unsafe_allow_html=True)
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    month_year = filtered_df.groupby(['month_added', 'year_added']).size().unstack().fillna(0)
    month_year = month_year.reindex(month_order)
    
    fig_heat = px.imshow(month_year, labels=dict(x="Year", y="Month", color="Titles Added"),
                         x=month_year.columns, y=month_year.index,
                         color_continuous_scale='Reds', template='plotly_dark',
                         title="Heatmap: Content Additions by Month & Year")
    fig_heat.update_layout(paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_heat, use_container_width=True)

# --- TAB 2: GEOGRAPHIC INTELLIGENCE ---
with tab2:
    st.markdown("<div class='section-header'>Global Content Distribution</div>", unsafe_allow_html=True)
    
    # Map
    country_counts = filtered_df[filtered_df['primary_country'] != 'Unknown']['primary_country'].value_counts().reset_index()
    country_counts.columns = ['Country', 'Total Titles']
    
    fig_map = px.choropleth(country_counts, locations="Country", locationmode="country names",
                            color="Total Titles", hover_name="Country", 
                            color_continuous_scale="Reds", title="Netflix Content by Country of Origin",
                            template='plotly_dark')
    fig_map.update_geos(showcountries=True, countrycolor="#333333", showcoastlines=False, showland=True, landcolor="#111111")
    fig_map.update_layout(paper_bgcolor='rgba(0,0,0,0)', geo_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig_map, use_container_width=True)
    
    colC, colD = st.columns(2)
    with colC:
        top_10_countries = country_counts.head(10)
        fig_bar_country = px.bar(top_10_countries, x='Total Titles', y='Country', orientation='h',
                                 title="Top 10 Producing Countries", color='Total Titles', color_continuous_scale='Reds',
                                 template='plotly_dark')
        fig_bar_country.update_layout(yaxis={'categoryorder':'total ascending'}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bar_country, use_container_width=True)
        
    with colD:
        st.markdown("""
        <div style="margin-top: 50px;" class='insight-box'>
            <strong>Executive Insight:</strong> The United States remains the absolute powerhouse for Netflix content generation, followed distantly by India and the UK. 
            <br><br>
            <strong>Actionable Recommendation:</strong> To capture emerging markets, aggressive investment in local originals from South Korea, Japan, and Latin America is recommended, as these regions show high engagement but lower overall catalog share.
        </div>
        """, unsafe_allow_html=True)

# --- TAB 3: GENRE & CONTENT ---
with tab3:
    st.markdown("<div class='section-header'>Genre Popularity & Ratings</div>", unsafe_allow_html=True)
    
    # Process Genres
    genres = filtered_df['listed_in'].str.split(', ').explode()
    genre_counts = genres.value_counts().head(15).reset_index()
    genre_counts.columns = ['Genre', 'Count']
    
    colE, colF = st.columns(2)
    with colE:
        fig_genre = px.treemap(genre_counts, path=['Genre'], values='Count',
                               title="Top 15 Most Popular Genres",
                               color='Count', color_continuous_scale='Reds')
        fig_genre.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_genre, use_container_width=True)
        
    with colF:
        rating_counts = filtered_df['rating'].value_counts().reset_index()
        rating_counts.columns = ['Rating', 'Count']
        fig_rating = px.bar(rating_counts, x='Rating', y='Count',
                            title="Content Distribution by Target Audience (Ratings)",
                            color_discrete_sequence=['#E50914'], template='plotly_dark')
        fig_rating.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_rating, use_container_width=True)
        
    # Duration Analysis
    st.markdown("<div class='section-header'>Duration Analysis (Movies)</div>", unsafe_allow_html=True)
    movies_df = filtered_df[filtered_df['type'] == 'Movie']
    
    fig_hist = px.histogram(movies_df, x='duration_num', nbins=50,
                            title="Distribution of Movie Durations (Minutes)",
                            color_discrete_sequence=['#E50914'], template='plotly_dark', marginal='box')
    fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("""
    <div class='insight-box'>
        <strong>Executive Insight:</strong> International Movies, Dramas, and Comedies dominate the catalog. The vast majority of content is mature (TV-MA), indicating Netflix's primary demographic leans adult. Movie durations are heavily clustered around 90-100 minutes, aligning with optimal viewer retention spans.
    </div>
    """, unsafe_allow_html=True)

# --- TAB 4: CAST & DIRECTORS ---
with tab4:
    st.markdown("<div class='section-header'>Top Talent: Directors & Actors</div>", unsafe_allow_html=True)
    
    colG, colH = st.columns(2)
    
    with colG:
        directors = filtered_df[filtered_df['director'] != 'Unknown']['director'].str.split(', ').explode()
        dir_counts = directors.value_counts().head(10).reset_index()
        dir_counts.columns = ['Director', 'Count']
        
        fig_dir = px.bar(dir_counts, x='Count', y='Director', orientation='h',
                         title="Top 10 Most Prolific Directors",
                         color_discrete_sequence=['#E50914'], template='plotly_dark')
        fig_dir.update_layout(yaxis={'categoryorder':'total ascending'}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_dir, use_container_width=True)
        
    with colH:
        cast = filtered_df[filtered_df['cast'] != 'Unknown']['cast'].str.split(', ').explode()
        cast_counts = cast.value_counts().head(10).reset_index()
        cast_counts.columns = ['Actor', 'Count']
        
        fig_cast = px.bar(cast_counts, x='Count', y='Actor', orientation='h',
                          title="Top 10 Most Prolific Actors",
                          color_discrete_sequence=['#ffffff'], template='plotly_dark')
        fig_cast.update_layout(yaxis={'categoryorder':'total ascending'}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_cast, use_container_width=True)
        
    st.markdown("""
    <div class='insight-box'>
        <strong>Executive Insight:</strong> International actors, particularly from Bollywood (Anupam Kher, Shah Rukh Khan), appear prominently at the top of the actors list due to the high volume of Indian cinema on the platform. This highlights the importance of the Indian market to Netflix's global strategy.
    </div>
    """, unsafe_allow_html=True)

# --- TAB 5: AI RECOMMENDATIONS ---
with tab5:
    st.markdown("<div class='section-header'>Machine Learning: Content Recommendation Engine</div>", unsafe_allow_html=True)
    st.markdown("Discover similar content based on natural language processing of descriptions, cast, directors, and genres.")
    
    # Ensure data for ML is clean
    ml_df = df.copy().reset_index(drop=True)
    
    # Recommendation UI
    selected_title = st.selectbox("Type or select a movie/TV show you like:", ml_df['title'].sort_values().tolist())
    
    if st.button("Generate Recommendations", type="primary"):
        with st.spinner('Running AI Recommendation Engine (TF-IDF & Cosine Similarity)...'):
            # Setup TF-IDF
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(ml_df['combined_features'])
            
            # Get index of selected title
            idx = ml_df[ml_df['title'] == selected_title].index[0]
            
            # Compute similarity for this specific title against all others
            cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
            
            # Get top 10 similar titles (excluding itself)
            similar_indices = cosine_sim.argsort()[:-12:-1]
            similar_indices = [i for i in similar_indices if i != idx][:10]
            
            recommendations = ml_df.iloc[similar_indices][['title', 'type', 'release_year', 'listed_in', 'description', 'primary_country']]
            recommendations['Similarity Score'] = cosine_sim[similar_indices]
            
        st.success(f"Top 10 recommendations based on '{selected_title}'")
        
        # Display recommendations in nice cards
        for _, row in recommendations.iterrows():
            st.markdown(f"""
            <div style="background-color: #222; padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #E50914;">
                <h4 style="margin: 0; color: #fff;">{row['title']} <span style="font-size: 14px; color: #aaa;">({row['release_year']}) - {row['type']}</span></h4>
                <p style="color: #E50914; font-size: 12px; margin-top: 5px; margin-bottom: 5px;"><strong>Match Score:</strong> {row['Similarity Score']:.2%} | <strong>Country:</strong> {row['primary_country']} | <strong>Genres:</strong> {row['listed_in']}</p>
                <p style="font-size: 14px; color: #ddd; margin: 0;">{row['description']}</p>
            </div>
            """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #aaa; padding: 20px;">
    <h3 style="color: #E50914; margin-bottom: 5px;">Shruti Singh</h3>
    <p style="font-size: 16px; margin-bottom: 5px;">Data Scientist • Machine Learning Engineer • AI Developer</p>
    <p style="font-size: 14px; margin-bottom: 20px; color: #888;">Turning Data into Actionable Intelligence through Analytics, Machine Learning, and Artificial Intelligence.</p>
    <p style="font-size: 12px; color: #555;">© 2026 Shruti Singh. All Rights Reserved.</p>
</div>
""", unsafe_allow_html=True)
