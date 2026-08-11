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

        response=ET.fromstring(response.content)
        articles=[]

        
