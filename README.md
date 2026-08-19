# ml-model
# Predictive Analytics Engine

## 📝 Description
The Predictive Analytics Engine integrates diverse data sources including financial market data, public interest metrics (Wikipedia views), and real-time news sentiment. It provides a robust machine learning pipeline for feature engineering and chronological model training, aiming to forecast trends and offer valuable insights based on combined historical and current information.

## 📂 Project Structure
```text
.
├── data_fetcher.py--fetches the data 
├── model_pipeline.py--main pipeline which handles the prediction
└── sentiment_engine.py--calculates the sentiment score between 0 and 1
```

## 🚀 Features
- Comprehensive Data Fetching: Retrieve historical stock data using `yfinance` and public interest trends via the Wikimedia page view API.,- Advanced Feature Engineering: Generate time-series specific features, including lagged values and various moving averages, for enhanced predictive modeling.,- Chronological Machine Learning Pipeline: Train and evaluate RandomForestRegressor models specifically designed for time-series data using a strict chronological split.,- Real-time News Sentiment Analysis: Fetch current news headlines from Google News RSS and perform sentiment analysis using NLTK's VADER lexicon.

## ⚙️ Installation & Setup
```bash
bash
git clone <repository_url>
cd <repository_name>
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
# NLTK's 'vader_lexicon' will be automatically downloaded by `sentiment_engine.py` if not found.
# As 'streamlit' is a dependency, you might run a Streamlit application if an `app.py` is present:
# streamlit run app.py

```
