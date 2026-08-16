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
    feature_cols=['Prev_Close','Prev_Volume','Prev_Wiki_Views','MA_7_Close','MA_14_Close']
    X= df[feature_cols]
    y= df['Target_Close']

    split_index= int(len(df)*train_ratio)
    X_train = X.iloc[:split_index]
    X_test  = X.iloc[split_index:]
    
    y_train = y.iloc[:split_index]
    y_test  = y.iloc[split_index:]

    model= RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train,y_train)

    predictions= model.predict(X_test)

    assert len(y_test) == len(predictions), f"Mismatch: y_test has {len(y_test)} rows, predictions has {len(predictions)}"

    rmse=np.sqrt(mean_squared_error(y_test, predictions))
    mape=mean_absolute_percentage_error(y_test, predictions) * 100

    metrics={
        "RMSE":round(rmse,2),
        "MAPE (%)":round(mape,2)
    }

    results_df= pd.DataFrame({
        'Actual_Close':y_test,
        'Predicted_Close':predictions
    }, index= y_test.index)

    return model,results_df,metrics,feature_cols

if __name__ == "__main__":
    test_ticker = "TSLA"
    test_wiki_title = "Tesla,_Inc."
    test_start = "2023-01-01"
    test_end = "2024-01-01"
    
    print("--- Fetching Raw Datasets ---")
    stock_df = fetch_stock_data(test_ticker, test_start, test_end)
    wiki_df = fetch_wikipedia_views(test_wiki_title, test_start, test_end)
    
    print("--- Building Feature Matrix & Eliminating Temporal Leakage ---")
    dataset = create_lagged_features(stock_df, wiki_df)
    print(f"Dataset ready with {len(dataset)} valid trading records.")
    
    print("\nSample Feature Row (X at t-1 predicting Y at t):")
    print(dataset[['Prev_Close', 'Prev_Wiki_Views', 'Target_Close']].head(3))
    
    print("\n--- Training Model & Evaluating ---")
    model, results_df, metrics, features = train_chronological_model(dataset)
    print(f"Evaluation Metrics: {metrics}")
    print("\nFirst 5 Predictions vs Actuals:")
    print(results_df.head(5))


    