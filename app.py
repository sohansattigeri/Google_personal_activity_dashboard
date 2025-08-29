import streamlit as st
import pandas as pd
import json
import plotly.express as px
from datetime import datetime
import openai
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Google Activity Dashboard", layout="wide")

st.title("🤖 AI-Enhanced Google My Activity Dashboard")
st.write("Upload your Google Takeout JSON to explore your activity with AI insights.")

# --- Sidebar for API Key ---
st.sidebar.header("🔑 API Key")
api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")
if api_key:
    openai.api_key = api_key

# --- File Upload ---
uploaded_file = st.file_uploader("Upload MyActivity.json", type="json")

if uploaded_file:
    data = json.load(uploaded_file)

    records = []
    for item in data:
        time = item.get("time")
        product = item.get("header", "Unknown")
        title = item.get("title", "")
        url = item.get("titleUrl", "")
        records.append([time, product, title, url])

    df = pd.DataFrame(records, columns=["time", "product", "title", "url"])
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])

    # Extract date parts
    df["date"] = df["time"].dt.date
    df["hour"] = df["time"].dt.hour
    df["day"] = df["time"].dt.day_name()

    # --- Quick Stats ---
    st.subheader("🔍 Quick Stats")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Unique Products", df["product"].nunique())
    col3.metric("Date Range", f"{df['date'].min()} → {df['date'].max()}")

    # --- Activity by Product ---
    st.subheader("📱 Activity by Google Product")
    product_count = df["product"].value_counts().reset_index()
    product_count.columns = ["Product", "Count"]
    fig1 = px.bar(product_count, x="Product", y="Count",
                  title="Activity by Google Product",
                  labels={"Product": "Google Product", "Count": "Activity Count"})
    st.plotly_chart(fig1, use_container_width=True)

    # --- Activity by Hour ---
    st.subheader("⏰ Activity by Hour of Day")
    fig2 = px.histogram(df, x="hour", nbins=24, title="Activity by Hour")
    st.plotly_chart(fig2, use_container_width=True)

    # --- Activity by Day of Week ---
    st.subheader("📅 Activity by Day of Week")
    fig3 = px.histogram(df, x="day",
                        category_orders={"day": ["Monday","Tuesday","Wednesday",
                                                 "Thursday","Friday","Saturday","Sunday"]},
                        title="Activity by Weekday")
    st.plotly_chart(fig3, use_container_width=True)

    # --- Word Cloud ---
    st.subheader("☁️ Word Cloud of Searches / Titles")
    text = " ".join(df["title"].dropna().tolist())
    if text:
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

    # --- AI Categorization ---
    if api_key:
        st.subheader("🧠 AI Activity Categorization")
        sample_titles = df["title"].dropna().head(20).tolist()
        prompt = f"""
        Categorize the following user activities into one of:
        - Learning
        - Work/Productivity
        - Entertainment
        - Social Media
        Activities: {sample_titles}
        Return results as a JSON list of {{activity, category}}.
        """
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content": prompt}]
            )
            st.json(response.choices[0].message.content)
        except Exception as e:
            st.error(f"AI categorization failed: {e}")

    # --- AI Weekly Summary ---
    if api_key:
        st.subheader("📖 AI Weekly Summary")
        week_data = df[df["date"] == df["date"].max()].head(50)[["product","title"]].to_string()
        summary_prompt = f"Summarize this activity data:\n{week_data}\nWrite it in simple English."
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are a personal activity analyst."},
                          {"role": "user", "content": summary_prompt}]
            )
            st.write(response.choices[0].message.content)
        except Exception as e:
            st.error(f"AI summary failed: {e}")

    # --- Raw Data ---
    with st.expander("📂 See Raw Data"):
        st.dataframe(df.head(100))

else:
    st.info("Please upload your `MyActivity.json` file from Google Takeout.")
# Note: To get your Google Takeout data, go to https://takeout.google.com/ and select "My Activity".