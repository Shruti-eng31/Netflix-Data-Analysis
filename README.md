# 🎥 Netflix Data Analysis & Insights Dashboard

![Netflix Logo](https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg)

An end-to-end Data Science and Business Intelligence portfolio project analyzing the global Netflix catalog. This project demonstrates data engineering, exploratory data analysis (EDA), natural language processing (NLP), machine learning (Recommendation Systems), and interactive UI development using Streamlit.

## 🚀 Project Overview
The objective of this project is to analyze Netflix's movies and TV shows dataset to uncover business insights, viewing trends, and geographical distribution. The final output is a production-quality Streamlit dashboard built for executive-level intelligence.

### 🌟 Key Features
*   **Comprehensive Data Cleaning:** Handled missing values, date conversions, string standardization, and numeric extractions.
*   **30+ Exploratory Visualizations:** Deep dive into global content distribution, genre analytics, country-wise production, rating heatmaps, and duration trends.
*   **AI Recommendation Engine:** Built using `scikit-learn` with **TF-IDF vectorization** and **Cosine Similarity** on combined text features (title, description, director, cast, genres).
*   **Premium Interactive Dashboard:** Built with Streamlit and custom CSS for a glassmorphism "Netflix Dark" theme.
*   **Business Intelligence:** Executive insights provided alongside every visualization to translate data into actionable recommendations.

---

## 🛠️ Technology Stack
*   **Language:** Python 3
*   **Data Manipulation:** Pandas, NumPy
*   **Visualization:** Matplotlib, Seaborn, Plotly Express, Plotly Graph Objects
*   **Machine Learning:** Scikit-Learn
*   **Dashboard Development:** Streamlit
*   **Styling:** HTML, CSS

---

## 📁 Project Structure

```text
Netflix-Data-Analysis/
│
├── data/                     # Data assets
│   └── netflix_titles.csv    # The Netflix Dataset
│
├── notebooks/                # Jupyter Notebooks
│   └── netflix_analysis.ipynb # Core data cleaning, EDA, and ML models
│
├── app/                      # Streamlit Application
│   └── streamlit_dashboard.py # Main interactive dashboard
│
├── requirements.txt          # Python dependencies
└── README.md                 # Project Documentation
```

---

## 🧠 Machine Learning Approach (Recommendation Engine)
To build a content-based recommendation system:
1.  **Feature Engineering:** A `combined_features` column was created by concatenating `title`, `director`, `cast`, `listed_in` (genres), and `description`.
2.  **Vectorization:** Used `TfidfVectorizer` (Term Frequency-Inverse Document Frequency) to convert the text data into numerical vectors, removing English stop words.
3.  **Similarity Matrix:** Calculated the `cosine_similarity` between all movies/shows based on their TF-IDF vectors.
4.  **Querying:** When a user selects a title, the engine finds the top 10 highest cosine similarity scores and returns the matching titles.

---

## 📊 Sample Business Insights
*   **Content Strategy:** Movies dominate the overall catalog (~70%), but TV Shows have seen a more aggressive growth rate in recent years as Netflix shifts focus to original series retention.
*   **Global Expansion:** The US is the absolute powerhouse for content generation. However, India is the second-largest contributor, heavily focused on Bollywood cinema, highlighting its importance to Netflix's global strategy.
*   **Audience Targeting:** The vast majority of content is mature (TV-MA). Over 60% of all content added in recent years caters strictly to adult audiences.

---

## ⚙️ Installation & Usage

1. **Clone the repository (or navigate to the project directory):**
   ```bash
   cd Netflix-Data-Analysis
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit Dashboard:**
   ```bash
   streamlit run app/streamlit_dashboard.py
   ```
   *The application will automatically open in your default web browser at `http://localhost:8501`.*

---

## 🔮 Future Improvements
*   **Time-Series Forecasting:** Implement ARIMA or Prophet to forecast future content acquisition rates.
*   **Sentiment Analysis:** Analyze descriptions to classify movies as 'dark/moody' vs. 'lighthearted'.
*   **User Collaborative Filtering:** Expand the ML engine using a mock user-ratings dataset to implement collaborative filtering.

*Designed for professional Data Science portfolios.*
