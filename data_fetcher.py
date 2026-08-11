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
    end_fmt= datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y%m%d00")
    url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
           f"en.wikipedia/all-access/user/{article_title}/daily/{start_fmt}/{end_fmt}")
    headers={'User-Agent': 'KassandraEngine/1.0 (contact@example.com)'}
    response= requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Warning: Failed to fetch Wikipedia data (Status Code: {response.status_code})")
        return pd.DataFrame(columns=['wiki_views'])
    data = response.json().get('items', [])
    if not data:
        return pd.DataFrame(columns=['wiki_views'])

    records=[]
    for item in data:
        date_str=item['timestamp'][:8]
        records.append({
            'Date': pd.to_datetime(date_str, format='%Y%m%d'),
            'wiki_views': item['views']
        })
    df= pd.DataFrame(records).set_index('Date')
    return df

if __name__ == "__main__":
    test_ticker = "TSLA"
    test_wiki_title = "Tesla,_Inc."
    test_start = "2023-01-01"
    test_end = "2024-01-01"
    
    print("--- Fetching Stock Data ---")
    stock_df = fetch_stock_data(test_ticker, test_start, test_end)
    print(stock_df.head(3))
    
    print("\n--- Fetching Wikipedia Pageviews ---")
    wiki_df = fetch_wikipedia_views(test_wiki_title, test_start, test_end)
    print(wiki_df.head(3))
