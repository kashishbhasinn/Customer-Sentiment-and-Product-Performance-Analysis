import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob

# Streamlit App Title
st.title("📊 Customer Sentiment and Product Performance Analysis")

# Upload Dataset
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Display Data Overview
    st.subheader("📌 Dataset Preview")
    st.write(df.head())

    # Ensure Required Columns Exist
    if "Product Names" not in df.columns or "Reviews" not in df.columns:
        st.error("❌ Dataset must contain 'Product Names' and 'Reviews' columns!")
    else:
        # Drop Missing Reviews
        df = df.dropna(subset=["Reviews"])

        # 📌 Identify Best-Selling Products
        st.subheader("🏆 Top 10 Best-Selling Products")

        best_selling = df.groupby("Product Names")["Reviews"].count().reset_index()
        best_selling = best_selling.sort_values(by="Reviews", ascending=False).head(10)

        # Plotly Bar Chart
        fig = px.bar(
            best_selling, 
            x="Reviews", 
            y="Product Names", 
            orientation="h",
            title="Top 10 Best-Selling Products",
            labels={"Reviews": "Number of Reviews", "Product Names": "Product"},
            color="Reviews",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig)

        # 📌 Sentiment Analysis
        st.subheader("📊 Sentiment Analysis of Reviews")

        def get_sentiment(review):
            sentiment = TextBlob(str(review)).sentiment.polarity
            if sentiment > 0:
                return "Positive"
            elif sentiment < 0:
                return "Negative"
            else:
                return "Neutral"

        df["Sentiment"] = df["Reviews"].apply(get_sentiment)
        sentiment_counts = df["Sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]

        # Plotly Pie Chart
        fig = px.pie(
            sentiment_counts, 
            names="Sentiment", 
            values="Count", 
            title="Sentiment Distribution",
            color="Sentiment",
            color_discrete_map={"Positive": "lightgreen", "Negative": "red", "Neutral": "gray"}
        )
        st.plotly_chart(fig)

        # 📌 Extract Most Frequent Words in Reviews
        st.subheader("🔍 Common Words in Reviews")
        all_reviews = " ".join(df["Reviews"].dropna()).lower()
        words = pd.Series(all_reviews.split()).value_counts().head(15)

        # Plotly Bar Chart for Most Used Words
        fig = px.bar(
            x=words.index, 
            y=words.values, 
            title="Most Frequently Used Words in Reviews",
            labels={"x": "Words", "y": "Frequency"},
            color=words.values,
            color_continuous_scale="teal"
        )
        st.plotly_chart(fig)

        # ✅ Success Message
        st.success("✅ Analysis Completed Successfully!")

else:
    st.warning("⚠️ Please upload a CSV file to proceed.")
