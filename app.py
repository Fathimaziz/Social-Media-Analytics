import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="BrandLens AI",
    page_icon="📊",
    layout="wide"
)

model = joblib.load(
    "models/sentiment_model.pkl"
)

vectorizer = joblib.load(
    "models/vectorizer.pkl"
)

st.title("📊 BrandLens AI")
st.subheader(
    "NLP Driven Social Media Analytics Dashboard"
)

menu = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Sentiment Prediction",
        "Word Cloud",
        "Engagement Analysis"
    ]
)

uploaded_file = st.file_uploader(
    "Upload Social Media Dataset",
    type=["csv"]
)

# -------------------------
# SENTIMENT PREDICTION
# -------------------------

if menu == "Sentiment Prediction":

    st.header("🔍 Live Sentiment Analyzer")

    user_text = st.text_area(
        "Enter Social Media Text"
    )

    if st.button("Analyze"):

        if user_text.strip() == "":
            st.warning("Please enter some text")

        else:

            transformed = vectorizer.transform(
                [user_text]
            )

            prediction = model.predict(
                transformed
            )[0]

            st.success(
                f"Predicted Sentiment: {prediction}"
            )

# -------------------------
# DATASET BASED ANALYSIS
# -------------------------

elif uploaded_file:

    df = pd.read_csv(uploaded_file)

    if menu == "Dashboard":

        total_posts = len(df)

        positive = len(
            df[df["Sentiment"] == "Positive"]
        )

        negative = len(
            df[df["Sentiment"] == "Negative"]
        )

        neutral = len(
            df[df["Sentiment"] == "Neutral"]
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Posts", total_posts)
        c2.metric("Positive", positive)
        c3.metric("Negative", negative)
        c4.metric("Neutral", neutral)

        fig = px.pie(
            df,
            names="Sentiment",
            title="Sentiment Distribution"
        )

        st.plotly_chart(fig)

        st.dataframe(df)

    elif menu == "Word Cloud":

        text = " ".join(df["Post"])

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white"
        ).generate(text)

        fig, ax = plt.subplots()

        ax.imshow(wordcloud)

        ax.axis("off")

        st.pyplot(fig)

    elif menu == "Engagement Analysis":

        fig1 = px.bar(
            df,
            x="Post",
            y="Likes",
            title="Likes Analysis"
        )

        st.plotly_chart(fig1)

        fig2 = px.bar(
            df,
            x="Post",
            y="Comments",
            title="Comments Analysis"
        )

        st.plotly_chart(fig2)

        fig3 = px.bar(
            df,
            x="Post",
            y="Shares",
            title="Shares Analysis"
        )

        st.plotly_chart(fig3)

else:

    st.info(
        "Upload a dataset to continue."
    )