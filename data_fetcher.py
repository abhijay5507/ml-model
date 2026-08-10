import pandas as pd
import yfinance as yf
import requests
from datetime import datetime

def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    df= yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        raise ValueError(f"No stock data found for ticker '{ticker}' between {start_date} and {end_date}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns= [col[0] for col in df.columns]
    df= df[['Close','Volume']].dropna()
    df.index= pd.to_datetime(df.index).tz_localize(None)
    df.index.name= 'Date'

    return df

def fetch_wikipedia_views(article_title: str, start_date: str, end_date:str) -> pd.DataFrame:
    start_fmt= datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y%m%d00")
    

if __name__ == "__main__":
    test_ticker = "TSLA"
    test_start = "2023-01-01"
    test_end = "2024-01-01"
    data = fetch_stock_data(test_ticker, test_start, test_end)
    print(f"Successfully fetched {len(data)} trading days for {test_ticker}:")
    print(data.head())
