import streamlit as st
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords

# Download necessary nltk resources
nltk.download('stopwords')

# Set Streamlit title
st.title("📊 Customer Sentiment and Product Performance Analysis")

# Upload dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Display dataset preview
    st.subheader("📌 Dataset Overview")
    st.write(df.head())

    # Handling missing values
    df = df.dropna(subset=["Reviews"])  

    # Identify Best-Selling Products
    st.subheader("🏆 Top 10 Best-Selling Products")

    best_selling = df.groupby("Product Names")["Reviews"].count().reset_index()
    best_selling = best_selling.sort_values(by="Reviews", ascending=False).head(10)

    # Bar Chart: Best-Selling Products
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(y=best_selling["Product Names"], x=best_selling["Reviews"], palette="Blues_r", ax=ax)
    ax.set_xlabel("Number of Reviews")
    ax.set_ylabel("Product Names")
    ax.set_title("Top 10 Best-Selling Products")
    st.pyplot(fig)

    # Sentiment Analysis Function
    def get_sentiment(review):
        sentiment = TextBlob(str(review)).sentiment.polarity
        if sentiment > 0:
            return "Positive"
        elif sentiment < 0:
            return "Negative"
        else:
            return "Neutral"

    df["Sentiment"] = df["Reviews"].apply(get_sentiment)

    # Sentiment Distribution
    st.subheader("📊 Sentiment Analysis of Reviews")

    sentiment_counts = df["Sentiment"].value_counts()

    fig, ax = plt.subplots()
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=["lightgreen", "lightcoral", "lightgray"])
    ax.set_title("Sentiment Distribution of Reviews")
    st.pyplot(fig)

    # Preprocess text for Word Cloud
    def preprocess_text(text):
        words = str(text).lower().split()
        words = [word for word in words if word not in stopwords.words('english')]  
        return " ".join(words)

    df["Cleaned_Reviews"] = df["Reviews"].apply(preprocess_text)

    # Extract Positive & Negative Reviews
    positive_reviews = " ".join(df[df["Sentiment"] == "Positive"]["Cleaned_Reviews"])
    negative_reviews = " ".join(df[df["Sentiment"] == "Negative"]["Cleaned_Reviews"])

    st.subheader("🔍 Most Used Words in Reviews")

    # Display Word Clouds
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Word Cloud for Positive Reviews
    wordcloud_pos = WordCloud(background_color="white", colormap="Greens", max_words=100).generate(positive_reviews)
    axes[0].imshow(wordcloud_pos, interpolation="bilinear")
    axes[0].axis("off")
    axes[0].set_title("Most Used Words in Positive Reviews")

    # Word Cloud for Negative Reviews
    wordcloud_neg = WordCloud(background_color="white", colormap="Reds", max_words=100).generate(negative_reviews)
    axes[1].imshow(wordcloud_neg, interpolation="bilinear")
    axes[1].axis("off")
    axes[1].set_title("Most Used Words in Negative Reviews")

    st.pyplot(fig)

    st.success("✅ Analysis Completed Successfully!")

else:
    st.warning("⚠️ Please upload a CSV file to proceed.")
