import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="NLP Review Analyzer", layout="wide")
st.title("🧠 NLP Review Analyzer")

@st.cache_data
def load():
    return pd.read_csv("data/reviews.csv")

df = load()

if {"sentiment", "topic"}.issubset(df.columns):

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Distribution")
        st.plotly_chart(
            px.histogram(df, x="sentiment", color="sentiment"),
            use_container_width=True,
        )

    with col2:
        st.subheader("Topic Frequencies")
        tc = df["topic"].value_counts().reset_index()
        tc.columns = ["Topic", "Count"]
        st.plotly_chart(
            px.bar(tc, x="Topic", y="Count", color="Topic"),
            use_container_width=True,
        )

    st.subheader("🔍 Raw Data")
    st.dataframe(df[["review", "sentiment", "topic"]], height=400)

else:
    st.warning("Run `nlp_pipeline.py` first to generate sentiment & topic columns.")
