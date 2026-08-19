import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Import custom backend modules
from data_fetcher import fetch_stock_data, fetch_wikipedia_views
from sentiment_engine import get_current_sentiment_score
from model_pipeline import create_lagged_features, train_chronological_model

# --- Page Configuration ---
st.set_page_config(page_title="Kassandra ML Engine", layout="wide")
st.title("📈 Kassandra ML: Market Prediction Engine")

# --- Sidebar Inputs ---
st.sidebar.header("Pipeline Parameters")
ticker = st.sidebar.text_input("Stock Ticker", value="TSLA").upper()
wiki_title = st.sidebar.text_input("Wikipedia Article", value="Tesla,_Inc.")

# Default date range: Last 365 days
end_date_default = datetime.today()
start_date_default = end_date_default - timedelta(days=365)

start_date = st.sidebar.date_input("Start Date", value=start_date_default).strftime("%Y-%m-%d")
end_date = st.sidebar.date_input("End Date", value=end_date_default).strftime("%Y-%m-%d")

# --- Run Pipeline Execution ---
if st.sidebar.button("Run ML Pipeline"):
    st.write(f"### 🚀 Executing AI Pipeline for {ticker}...")
    
    # 1. Fetch Live Sentiment
    with st.spinner("Fetching live news sentiment..."):
        sentiment_score = get_current_sentiment_score(ticker)
        st.metric(label="Real-Time News Sentiment (VADER)", value=f"{sentiment_score} / 1.0")
        
    # 2. Fetch Market & Alternative Data
    with st.spinner("Downloading Market & Wikipedia Data..."):
        stock_df = fetch_stock_data(ticker, start_date, end_date)
        wiki_df = fetch_wikipedia_views(wiki_title, start_date, end_date)
        
    # 3. Process Features & Train Model
    with st.spinner("Aligning Temporal Features & Training Model..."):
        dataset = create_lagged_features(stock_df, wiki_df)
        model, results_df, metrics, features = train_chronological_model(dataset)
        
    st.success("Pipeline executed successfully!")
    
    # --- NEW: DEDICATED PREDICTION CARDS SECTION ---
    st.write("### 🎯 Single-Price Predictions & Live Forecast")
    
    # Extract latest test row values
    latest_date = results_df.index[-1].strftime("%Y-%m-%d")
    latest_actual = results_df['Actual_Close'].iloc[-1]
    latest_predicted = results_df['Predicted_Close'].iloc[-1]
    pred_delta = latest_predicted - latest_actual

    # Compute Next-Day Forecast (predicting tomorrow using today's latest features)
    latest_feature_row = dataset[features].tail(1)
    next_day_forecast = model.predict(latest_feature_row)[0]

    p_col1, p_col2, p_col3 = st.columns(3)
    p_col1.metric(
        label=f"Actual Close ({latest_date})", 
        value=f"${latest_actual:.2f}"
    )
    p_col2.metric(
        label=f"Predicted Close ({latest_date})", 
        value=f"${latest_predicted:.2f}", 
        delta=f"${pred_delta:.2f}"
    )
    p_col3.metric(
        label="⚡ Next-Day Live Forecast (t+1)", 
        value=f"${next_day_forecast:.2f}"
    )
    
    st.write("---")
    
    # --- Model Evaluation Metrics ---
    st.write("### 📊 Model Performance Metrics")
    col1, col2 = st.columns(2)
    col1.metric("Model RMSE (Root Mean Squared Error)", f"${metrics['RMSE']}")
    col2.metric("Model MAPE (Mean Absolute Percentage Error)", f"{metrics['MAPE (%)']}%")
    
    # --- Interactive Chart ---
    st.write("### 📈 Actual vs Predicted Closing Prices (Unseen Test Set)")
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=results_df.index, y=results_df['Actual_Close'], 
        mode='lines+markers', name='Actual Price', line=dict(color='#1f77b4', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=results_df.index, y=results_df['Predicted_Close'], 
        mode='lines+markers', name='Predicted Price', line=dict(color='#ff7f0e', dash='dash')
    ))
    
    fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    
    # --- Expandable Raw Data View ---
    with st.expander("🔍 View Raw Feature Matrix (Last 10 Days)"):
        st.dataframe(dataset.tail(10))