"""
Step 3: Building Streamlit Application for Interactive Dashboard
----------------------------------------
This script creates an interactive dashboard to analyze the "Student Habits & Performance" dataset. 
The dashboard allows users to filter the dataset by different categorical variables, view key metrics, 
and explore multiple visualizations including boxplots, scatter plots, bar charts, and correlation heatmaps.

Features:
---------
Sidebar filters for gender, part-time job, diet quality, internet quality, extracurricular participation, and parental education  
Key metrics (average study hours, sleep hours, exam score)  
Interactive visualizations with Plotly, Seaborn, and Streamlit’s built-in charts  
Correlation heatmap for numerical features  
----------------------------------------

"""

# ----------------------------
# Import Libraries
# ----------------------------
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Student Habits & Performance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Load Dataset
# ----------------------------
@st.cache_data
def load_data():
    """
    Load the student habits & performance dataset.
    
    Returns
    -------
    pd.DataFrame
        Dataframe containing student lifestyle habits and performance scores.
    """
    df = pd.read_csv("data/student_habits_performance.csv")
    return df

df = load_data()

# ----------------------------
# Dashboard Title & Intro
# ----------------------------
st.title("📊 Student Habits & Performance Dashboard")
st.markdown("""
Welcome to the interactive dashboard!  
Explore how lifestyle habits, environment, and study patterns influence student exam performance.  
Use the filters on the left to slice the dataset and view updated metrics and visualizations.
""")

# ----------------------------
# Sidebar: Filters
# ----------------------------
st.sidebar.header("🔍 Filter Panel")

# Gender filter
gender_filter = st.sidebar.radio(
    "Select Gender", options=["All"] + list(df["gender"].unique())
)

# Part-time job filter
job_filter = st.sidebar.multiselect(
    "Part-time Job", options=df["part_time_job"].unique(),
    default=df["part_time_job"].unique()
)

# Diet quality filter
diet_filter = st.sidebar.multiselect(
    "Diet Quality", options=df["diet_quality"].unique(),
    default=df["diet_quality"].unique()
)

# Internet quality filter
internet_filter = st.sidebar.multiselect(
    "Internet Quality", options=df["internet_quality"].unique(),
    default=df["internet_quality"].unique()
)

# Extracurricular participation filter
extra_filter = st.sidebar.multiselect(
    "Extracurricular Activities", options=df["extracurricular_participation"].unique(),
    default=df["extracurricular_participation"].unique()
)

# Parental education filter
edu_filter = st.sidebar.multiselect(
    "Parental Education Level", 
    options=df["parental_education_level"].dropna().unique(),
    default=df["parental_education_level"].dropna().unique()
)

# ----------------------------
# Apply Filters
# ----------------------------
filtered_df = df.copy()

if gender_filter != "All":
    filtered_df = filtered_df[filtered_df["gender"] == gender_filter]

if job_filter:
    filtered_df = filtered_df[filtered_df["part_time_job"].isin(job_filter)]

if diet_filter:
    filtered_df = filtered_df[filtered_df["diet_quality"].isin(diet_filter)]

if internet_filter:
    filtered_df = filtered_df[filtered_df["internet_quality"].isin(internet_filter)]

if extra_filter:
    filtered_df = filtered_df[filtered_df["extracurricular_participation"].isin(extra_filter)]

if edu_filter:
    filtered_df = filtered_df[filtered_df["parental_education_level"].isin(edu_filter)]

# ----------------------------
# Key Metrics
# ----------------------------
st.markdown("### 📌 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric(
    "📈 Avg Study Hours/Day", 
    f"{filtered_df['study_hours_per_day'].mean():.2f}"
)
col2.metric(
    "🛏 Avg Sleep Hours", 
    f"{filtered_df['sleep_hours'].mean():.2f}"
)
col3.metric(
    "🎯 Avg Exam Score", 
    f"{filtered_df['exam_score'].mean():.2f}"
)

# ----------------------------
# Visualization 1: Boxplot
# ----------------------------
st.subheader("📊 Exam Score Distribution by Gender")
fig = px.box(
    filtered_df, 
    x="gender", y="exam_score", color="gender", 
    title="Exam Scores by Gender"
)
st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Visualization 2: Scatter Plot
# ----------------------------
st.subheader("📊 Study Hours vs Exam Score (Mental Health Rating)")
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(
    data=filtered_df, 
    x="study_hours_per_day", 
    y="exam_score", 
    hue="mental_health_rating", 
    palette="viridis", 
    ax=ax
)
plt.title("Study Hours vs Exam Score (colored by Mental Health Rating)")
st.pyplot(fig)

# ----------------------------
# Visualization 3: Bar Chart
# ----------------------------
st.subheader("📊 Average Exam Score by Diet Quality")
avg_scores = filtered_df.groupby("diet_quality")["exam_score"].mean()
st.bar_chart(avg_scores)

# ----------------------------
# Visualization 4: Correlation Heatmap
# ----------------------------
st.subheader("📊 Correlation Heatmap of Numerical Features")
corr = filtered_df.select_dtypes(include=["float64", "int64"]).corr()

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(
    corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax
)
plt.title("Correlation Heatmap")
st.pyplot(fig)