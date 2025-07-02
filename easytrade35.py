import yfinance as yf
import finnhub
import feedparser
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# API Keys
FINNHUB_API_KEY = 'YOUR_FINNHUB_API_KEY'
QUIVER_API_KEY = '9ed914d4d32da4d26b02d4d9540f46606002736b'

# Initialize API clients
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
analyzer = SentimentIntensityAnalyzer()

stocks = {
    'Top_Tier': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
    'Mid_Tier': ['SOFI', 'NXT', 'MGNI', 'JOBY', 'SKYE'],
    'Political': ['BA', 'LMT', 'RTX', 'NOC'],
    'EV': ['TSLA', 'LCID', 'RIVN', 'NIO'],
    'Energy': ['XOM', 'CVX', 'SLB', 'BP'],
    'Tech': ['AMD', 'PLTR', 'INTC', 'CRM', 'ORCL']
}

stock_list = [stock for sector in stocks.values() for stock in sector]

def fetch_news_sentiment(ticker):
    try:
        feed = feedparser.parse(f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US")
        sentiment_score = 0
        count = 0
        for entry in feed.entries[:10]:
            title = entry.get('title', '')
            summary = entry.get('summary', '')
            text = (title + ' ' + summary).lower()
            sentiment = analyzer.polarity_scores(text)
            if 'upgrade' in text or 'beats' in text:
                sentiment['compound'] += 0.1
            elif 'downgrade' in text or 'misses' in text:
                sentiment['compound'] -= 0.1
            sentiment_score += sentiment['compound']
            count += 1
        return sentiment_score / count if count > 0 else 0
    except Exception as e:
        print(f"Error fetching Yahoo news for {ticker}: {e}")
        return 0

def fetch_social_sentiment(ticker):
    return 0  # Temporarily disabled

def fetch_insider_trading(ticker):
    try:
        headers = {"Authorization": f"Bearer {QUIVER_API_KEY}"}
        url = "https://api.quiverquant.com/beta/live/congresstrading"
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"Quiver API error: {response.status_code}")
            return 0

        data = response.json()
        recent_trades = [t for t in data if t.get("Ticker", "").upper() == ticker.upper()]
        recent_trades = recent_trades[:10]

        score = 0
        for trade in recent_trades:
            action = trade.get("Transaction", "").lower()
            amount = trade.get("Amount", 0)
            rep_name = trade.get("Representative", "").lower()

            if "purchase" in action or "buy" in action:
                score += 0.2 + min(amount / 100000, 0.3)
            elif "sale" in action or "sell" in action:
                score -= 0.2 + min(amount / 100000, 0.3)

            if "pelosi" in rep_name:
                score += 0.3

        print(f"{ticker} insider score: {score}")
        return score
    except Exception as e:
        print(f"Error fetching insider data for {ticker}: {e}")
        return 0

                data = response.json()
        recent_trades = [t for t in data if t.get("Ticker", "").upper() == ticker.upper()]
        recent_trades = recent_trades[:10]

        score = 0
        for trade in recent_trades:
            action = trade.get("Transaction", "").lower()
            if "purchase" in action or "buy" in action:
                score += 0.2
            elif "sale" in action or "sell" in action:
                score -= 0.2

        print(f"{ticker} insider score: {score}")
        return score
    except Exception as e:
        print(f"Error fetching insider data for {ticker}: {e}")
        return 0

        data = response.json()
        recent_trades = [t for t in data if t.get("Ticker", "").upper() == ticker.upper()]
        recent_trades = recent_trades[:10]

        score = 0
        for trade in recent_trades:
            action = trade.get("Transaction", "").lower()
            if "buy" in action:
                score += 0.2
            elif "sell" in action:
                score -= 0.2

        print(f"{ticker} insider score: {score}")
        return score
    except Exception as e:
        print(f"Error fetching insider data for {ticker}: {e}")
        return 0
