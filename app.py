import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from data_fetcher import fetch_stock_data, fetch_wikipedia_views
from sentiment_engine import get_current_sentiment_score
from model_pipeline import create_lagged_features, train_chronological_model

st.set_page_config(page_title="Kassandra ML Engine", layout="wide")
st.title("📈 Kassandra ML: Market Prediction Engine")

st.sidebar.header("Pipeline Parameters")
ticker = st.sidebar.text_input("Stock Ticker", value="TSLA").upper()
wiki_title = st.sidebar.text_input("Wikipedia Article", value="Tesla,_Inc.")

end_date_default = datetime.today()
start_date_default = end_date_default - timedelta(days=365)

start_date = st.sidebar.date_input("Start Date", value=start_date_default).strftime("%Y-%m-%d")
end_date = st.sidebar.date_input("End Date", value=end_date_default).strftime("%Y-%m-%d")

if st.sidebar.button("Run ML Pipeline"):
    st.write(f"### 🚀 Executing AI Pipeline for {ticker}...")
    
    with st.spinner("Fetching live news sentiment..."):
        sentiment_score = get_current_sentiment_score(ticker)
        st.metric(label=f"Real-Time News Sentiment (VADER)", value=f"{sentiment_score} / 1.0")
        
    with st.spinner("Downloading Market & Wikipedia Data..."):
        stock_df = fetch_stock_data(ticker, start_date, end_date)
        wiki_df = fetch_wikipedia_views(wiki_title, start_date, end_date)
        
    with st.spinner("Aligning Temporal Features & Training Model..."):
        dataset = create_lagged_features(stock_df, wiki_df)
        model, results_df, metrics, features = train_chronological_model(dataset)
        
    st.success("Pipeline executed successfully!")
    
    col1, col2 = st.columns(2)
    col1.metric("Model RMSE (Root Mean Squared Error)", f"${metrics['RMSE']}")
    col2.metric("Model MAPE (Mean Absolute Percentage Error)", f"{metrics['MAPE (%)']}%")
    
    st.write("### 📊 Actual vs Predicted Closing Prices (Unseen Test Set)")
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=results_df.index, y=results_df['Actual_Close'], 
        mode='lines', name='Actual Price', line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=results_df.index, y=results_df['Predicted_Close'], 
        mode='lines', name='Predicted Price', line=dict(color='orange', dash='dash')
    ))
    
    fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("🔍 View Raw Feature Matrix (Last 10 Days)"):
        st.dataframe(dataset.tail(10))