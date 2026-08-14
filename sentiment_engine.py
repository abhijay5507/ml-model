import xml.etree.ElementTree as ET
from urllib.parse import quote_plus
from datetime import datetime
import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

def fetch_news_headlines(query: str, max_results: int) -> list[dict]:
    encoded_query= quote_plus(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    try:
        response= requests.get(rss_url, headers=headers, timeout=10)
        if response.status_code !=200:
            return[]

        root=ET.fromstring(response.content)
        articles=[]
        for item in root.findall('./channel/item')[:max_results]:
            title = item.find('title')
            pub_date= item.find('pubDate')

            if title is not None and title.text:
                articles.append({
                    "title":title.text,
                    "pub_date": pub_date.text if pub_date is not None else ""
                })
        return articles

    except Exception as e:
        print(f"Error fetching news for {query}: {e}")
        return []

def analyze_sentiment(texts: list[str]) -> dict:

    if not texts:
        return{
            "compound": 0.0,
            "pos": 0.0,
            "neg": 0.0,
            "neu": 1.0,
            "label": "Neutral"
        }

    sia= SentimentIntensityAnalyzer()
    compound_scores=[]
    pos_scores=[]
    neg_scores=[]
    neu_scores=[]

    for text in texts:
        scores= sia.polarity_scores(text)
        compound_scores.append(scores['compound'])
        pos_scores.append(scores['pos'])
        neg_scores.append(scores['neg'])
        neu_scores.append(scores['neu'])
    avg_compound= float(sum(compound_scores)/len(compound_scores))

    if avg_compound>=0.05:
        label= "Bullish/Positive"
    elif avg_compound<=-0.05:
        label="Bearish/Negative"
    else:
        label="Neutral"

    return{"compound": round(avg_compound, 4),
        "pos": round(float(sum(pos_scores) / len(pos_scores)), 4),
        "neg": round(float(sum(neg_scores) / len(neg_scores)), 4),
        "neu": round(float(sum(neu_scores) / len(neu_scores)), 4),
        "label": label
    }

def get_current_sentiment_score(query: str)->float:
    articles= fetch_news_headlines(query, max_results=15)
    titles=[a["title"] for a in articles]
    sentiment_data= analyze_sentiment(titles)
    return sentiment_data["compound"]

if __name__ == "__main__":
    test_query = "Tesla"
    print(f"--- Fetching Live News & Sentiment for: {test_query} ---")
    
    news_items = fetch_news_headlines(test_query, max_results=5)
    print(f"\nFound {len(news_items)} headlines:")
    for idx, item in enumerate(news_items, 1):
        print(f"{idx}. {item['title']}")
        
    titles_only = [item['title'] for item in news_items]
    sentiment_result = analyze_sentiment(titles_only)
    
    print("\n--- Sentiment Analysis Output ---")
    print(f"Compound Score: {sentiment_result['compound']}")
    print(f"Sentiment Label: {sentiment_result['label']}")
    print(f"Positive Ratio: {sentiment_result['pos']}")
    print(f"Negative Ratio: {sentiment_result['neg']}")