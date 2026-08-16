import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error 
from data_fetcher import fetch_stock_data, fetch_wikipedia_views

def create_lagged_features(stock_df: pd.DataFrame, wiki_df: pd.DataFrame) -> pd.DataFrame:

    merged_df= stock_df.join(wiki_df, how='left')
    merged_df['wiki_views']=merged_df['wiki_views'].ffill().fillna(0)
    merged_df['Target_Close']=merged_df['Close']

    merged_df['Prev_Close']=merged_df['Close'].shift(1)
    merged_df['Prev_Volume']=merged_df['Volume'].shift(1)
    merged_df['Prev_Wiki_Views']=merged_df['wiki_views'].shift(1)

    merged_df['MA_7_Close'] = merged_df['Prev_Close'].rolling(window=7).mean()
    merged_df['MA_14_Close'] = merged_df['Prev_Close'].rolling(window=14).mean()

    clean_df= merged_df.dropna().copy()

    return clean_df

def train_chronological_model(df: pd.DataFrame,train_ratio: float=0.8):
    feature_cols=['Prev_Close','Prev_Volumes','Prev_Wiki_Views','MA_7_Close','MA_14_Close']
    X= df[feature_cols]
    y= df['Target_Close']

    split_index= int(len(df)*train_ratio)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    
