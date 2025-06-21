
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import base64
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="📊 Sentiment & Stock Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- App Header ---
st.markdown("""
    <style>
    .css-1aumxhk {
        padding: 2rem;
        background-color: #f7f4e9;
        border-radius: 1rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }
    .stButton > button {
        font-weight: bold;
        background-color: #ffcf33;
        color: #333;
        border-radius: 10px;
        padding: 10px 20px;
        margin: 5px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📈 Social Media Sentiment + Stock Price Dashboard")
st.caption("Analyze, Predict, and Strategize with Sentiment and Market Data")

# --- Sidebar Inputs ---
st.sidebar.header("Upload Datasets")
social_file = st.sidebar.file_uploader("Upload Social Media CSV", type="csv")
stock_file = st.sidebar.file_uploader("Upload Stock Market CSV", type="csv")

# --- Theme Toggle ---
selected_theme = st.sidebar.radio("Choose Theme", ["Light", "Dark"])
if selected_theme == "Dark":
    st.markdown("""<style>body { background-color: #1e1e1e; color: white; }</style>""", unsafe_allow_html=True)

# --- Load Data ---
social_df, stock_df = None, None
if social_file:
    social_df = pd.read_csv(social_file)
    st.success(f"Social data loaded: {len(social_df)} records")
if stock_file:
    stock_df = pd.read_csv(stock_file)
    st.success(f"Stock data loaded: {len(stock_df)} records")

# --- Action Buttons ---
cols = st.columns(5)
run_diag = cols[0].button("📊 Diagnostic Analysis")
run_pred = cols[1].button("🔮 Predict Next-Day Price")
run_presc = cols[2].button("📈 Prescriptive Analysis")
run_perf = cols[3].button("📉 Performance Analysis")
cols[4].download_button("📥 Download Sample Chart", "This is a placeholder.", file_name="chart.txt")

# --- Plotting Section ---
if stock_df is not None:
    st.markdown("---")
    st.subheader("📈 Sample Stock Trend")
    stock_df.columns = stock_df.columns.str.strip().str.lower()
    if 'date' in stock_df.columns and 'close' in stock_df.columns:
        stock_df['date'] = pd.to_datetime(stock_df['date'], errors='coerce')
        stock_df.dropna(subset=['date'], inplace=True)
        fig = px.line(stock_df, x='date', y='close', title="Stock Price Over Time", template="plotly_dark" if selected_theme == "Dark" else "plotly")
        st.plotly_chart(fig, use_container_width=True)

# --- Diagnostic Analysis ---
if run_diag and social_df is not None and stock_df is not None:
    st.markdown("---")
    st.subheader("📊 Sentiment Distribution and Correlation")
    try:
        social_df['date'] = pd.to_datetime(social_df['date'], errors='coerce')
        social_df.dropna(subset=['text', 'date'], inplace=True)
        social_df['sentiment_score'] = social_df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        social_df['sentiment_category'] = social_df['sentiment_score'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))
        sentiment_counts = social_df['sentiment_category'].value_counts()

        fig_pie = px.pie(
            names=sentiment_counts.index,
            values=sentiment_counts.values,
            title="Sentiment Distribution",
            color_discrete_sequence=["#22eeaa", "#8888aa", "#ff5566"]
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        sentiment_trend = social_df.groupby(social_df['date'].dt.date)['sentiment_score'].mean().reset_index()
        stock_df['date'] = pd.to_datetime(stock_df['date'], errors='coerce')
        stock_df['date'] = stock_df['date'].dt.date
        merged = pd.merge(sentiment_trend, stock_df[['date', 'close']], on='date', how='inner')

        fig_corr = px.line(merged, x='date', y=['sentiment_score', 'close'], title="Average Sentiment vs Close Price")
        st.plotly_chart(fig_corr, use_container_width=True)
    except Exception as e:
        st.error(f"Error in Diagnostic Analysis: {e}")

# --- Predict Next-Day Price ---
if run_pred and social_df is not None and stock_df is not None:
    st.markdown("---")
    st.subheader("🔮 Predicting Next-Day Stock Price")
    try:
        social_df['date'] = pd.to_datetime(social_df['date'], errors='coerce')
        social_df['sentiment_score'] = social_df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        sentiment_trend = social_df.groupby(social_df['date'].dt.date)['sentiment_score'].mean().reset_index()

        stock_df['date'] = pd.to_datetime(stock_df['date'], errors='coerce')
        stock_df['date'] = stock_df['date'].dt.date
        data = pd.merge(stock_df[['date', 'close']], sentiment_trend, on='date', how='inner')
        data = data.sort_values('date')
        data['next_close'] = data['close'].shift(-1)
        data.dropna(inplace=True)
        data['price_diff'] = data['next_close'] - data['close']

        X = data[['close', 'sentiment_score']]
        y = data['price_diff']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        st.write(f"Mean Squared Error: {mse:.4f}")

        fig_pred = px.line(y_test.reset_index(drop=True), labels={'value': 'Price Diff'}, title="Actual vs Predicted Price Movement")
        fig_pred.add_scatter(y=y_pred, mode='lines', name='Predicted')
        st.plotly_chart(fig_pred, use_container_width=True)
    except Exception as e:
        st.error(f"Error in Prediction: {e}")

# --- Prescriptive Analysis ---
if run_presc and social_df is not None and stock_df is not None:
    st.markdown("---")
    st.subheader("📈 Prescriptive Strategy Recommendation")
    try:
        social_df['date'] = pd.to_datetime(social_df['date'], errors='coerce')
        social_df['sentiment_score'] = social_df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        sentiment_trend = social_df.groupby(social_df['date'].dt.date)['sentiment_score'].mean().reset_index()

        stock_df['date'] = pd.to_datetime(stock_df['date'], errors='coerce')
        stock_df['date'] = stock_df['date'].dt.date
        merged = pd.merge(sentiment_trend, stock_df[['date', 'close']], on='date', how='inner')
        merged['price_diff'] = merged['close'].diff()
        merged.dropna(inplace=True)

        merged['buy_signal'] = (merged['avg_sentiment'] > 0.5) & (merged['price_diff'] < 0)
        buy_dates = merged[merged['buy_signal']]['date'].tolist()

        st.write(f"Buy signals (Sentiment > 0.5 & Price Drop): {len(buy_dates)}")
        st.write("Sample Dates:", buy_dates[:10])

        negative_sentiment = merged[merged['avg_sentiment'] < 0]
        if not negative_sentiment.empty:
            corr_neg = negative_sentiment['avg_sentiment'].corr(negative_sentiment['price_diff'])
            if corr_neg < -0.3:
                st.info("Marketing Insight: Strong negative sentiment correlates with price drops. Avoid product announcements during this time.")
            else:
                st.info("Marketing Insight: No strong negative correlation. Timing less critical.")
        else:
            st.warning("Not enough negative sentiment data for marketing recommendation.")
    except Exception as e:
        st.error(f"Error in Prescriptive Analysis: {e}")

# --- Footer ---
st.markdown("""
<hr style="margin-top:3rem;">
<p style="text-align: center; font-size: 0.9rem;">Made with ❤️ using Streamlit</p>
""", unsafe_allow_html=True)
