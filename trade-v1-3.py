# Option Strategy Recommender - Elite Trader Version with Enhanced Put/Call Balance

import yfinance as yf
from datetime import datetime, timedelta
from newsapi import NewsApiClient
import requests
from bs4 import BeautifulSoup
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Constants
NEWS_API_KEY = 'your_newsapi_key_here'
THRESHOLD_SCORE = 60
LOOKAHEAD_DAYS = 30

# Initialize APIs
newsapi = NewsApiClient(api_key=NEWS_API_KEY)
sentiment_analyzer = SentimentIntensityAnalyzer()

# Weights
WEIGHTS = {
    'trend': 15,
    'volume': 10,
    'news_sentiment': 15,
    'earnings': 10,
    'sector_strength': 15,
    'insider': 15,
    'peg': 10,
    'fcf': 5,
    'roic': 5
}

# Expanded ticker list including large-cap, mid-cap, growth, and emerging stocks
TICKER_LIST = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'UNH', 'AVGO', 'LLY', 'XOM',
    'JNJ', 'V', 'WMT', 'MA', 'CVX', 'ABBV', 'MRK', 'HD', 'KO', 'PEP', 'ORCL', 'CRM', 'COST',
    'SMCI', 'PLTR', 'AMD', 'INTC', 'NFLX', 'UBER', 'PATH', 'UPST', 'NET', 'DDOG', 'ZS',
    'SNOW', 'RBLX', 'TWLO', 'ROKU', 'DOCN', 'AI', 'ENVX', 'FSLR', 'RUN', 'SPWR',
    'TSLA', 'LCID', 'RIVN', 'NIO', 'CHPT', 'BLNK', 'EVGO', 'WBX', 'FREY', 'LTHM',
    'ASTS', 'SOUN', 'BBAI', 'RKLB', 'DNA', 'IONQ', 'GTLB', 'MNMD', 'CRSP', 'EDIT',
    'VIR', 'ICPT', 'RNA', 'VRTX', 'REGN', 'BIIB', 'ARWR', 'EXEL', 'SGEN', 'ILMN',
    'XLE', 'RIG', 'HAL', 'SD', 'SM', 'LMT', 'NOC', 'BA', 'GD', 'RTX',
    'NANC', 'QQQ', 'SPY', 'ARKK', 'IWM', 'VTI', 'VOO', 'XLK', 'XLF', 'XLY'
]

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="6mo")
    info = stock.info
    return hist, info

def analyze_trend(hist):
    if len(hist) < 20:
        return 0
    return int(hist['Close'].iloc[-1] > hist['Close'].mean())

def analyze_volume(hist):
    return int(hist['Volume'].iloc[-1] > hist['Volume'].rolling(20).mean().iloc[-1])

def analyze_news_sentiment(ticker):
    try:
        articles = newsapi.get_everything(q=ticker, language='en', sort_by='relevancy', page_size=10)
        score = 0
        for a in articles['articles']:
            score += sentiment_analyzer.polarity_scores(a['title'])['compound']
        return max(min(score * 10, 100), -100)
    except:
        return 0

def has_upcoming_earnings(info):
    earnings_date = info.get('earningsDate')
    if isinstance(earnings_date, list):
        earnings_date = earnings_date[0]
    if earnings_date:
        try:
            days_until = (earnings_date - datetime.today()).days
            return int(days_until < LOOKAHEAD_DAYS)
        except:
            return 0
    return 0

def get_insider_score(ticker):
    return 10  # Placeholder for real insider trading logic

def get_peg_score(info):
    peg = info.get('pegRatio')
    return 10 if peg and peg < 1.5 else 0

def get_fcf_score(info):
    fcf = info.get('freeCashflow')
    return 5 if fcf and fcf > 0 else 0

def get_roic_score(info):
    roic = info.get('returnOnEquity')
    return 5 if roic and roic > 0.15 else 0

def compute_total_score(features):
    return sum(features[k] * WEIGHTS[k] / 10 for k in WEIGHTS)

def estimate_earnings_potential(strategy, price):
    if strategy == 'Buy Stock':
        return f"Potential Gain: ~10-15% in 3–6 months (${price * 0.10:.2f}–${price * 0.15:.2f})"
    elif strategy == 'Bull Call Spread':
        return "Max Profit: ~$350 per contract (risk ~$150)"
    elif strategy == 'Naked Call':
        return "High reward, high risk – potential 100%+ return on premium"
    elif strategy == 'Diagonal Call Spread':
        return "Return: ~25–40% if short leg expires worthless"
    elif strategy == 'Cash-Secured Put':
        return "Yield: ~1.5–2% in 30 days (~18–24% annualized)"
    else:
        return "N/A"

def recommend_strategy(score, info):
    peg = info.get('pegRatio', 2)
    fcf = info.get('freeCashflow', 0)
    if score >= 85 and fcf > 0 and peg < 1.5:
        return 'Buy Stock'
    elif score >= 85 and (fcf < 0 or peg > 2):
        return 'Cash-Secured Put'
    elif score > 85:
        return 'Naked Call'
    elif score > 75:
        return 'Bull Call Spread'
    elif score > 65:
        return 'Diagonal Call Spread'
    elif score > 50:
        return 'Cash-Secured Put'
    else:
        return 'Avoid - No Trade'

def suggest_trade_details(ticker, hist, rec_type):
    current_price = hist['Close'].iloc[-1]
    expiration = datetime.now() + timedelta(days=30)
    base = round(current_price / 5) * 5
    if rec_type == 'Buy Stock':
        return f"Buy {ticker} shares at ${current_price:.2f}", estimate_earnings_potential(rec_type, current_price)
    elif rec_type == 'Bull Call Spread':
        return f"Buy {base} Call / Sell {base + 5} Call expiring {expiration.date()} at limit ~1.5", estimate_earnings_potential(rec_type, current_price)
    elif rec_type == 'Naked Call':
        return f"Buy {base} Call expiring {expiration.date()} at limit ~2.0", estimate_earnings_potential(rec_type, current_price)
    elif rec_type == 'Diagonal Call Spread':
        return f"Buy {base} Call (45d) / Sell {base + 5} Call (15d)", estimate_earnings_potential(rec_type, current_price)
    elif rec_type == 'Cash-Secured Put':
        return f"Sell {base - 5} Put expiring {expiration.date()} at credit ~1.2 (hold $100/share cash)", estimate_earnings_potential(rec_type, current_price)
    else:
        return "No trade recommended", "N/A"

def run_analysis(ticker):
    hist, info = get_stock_data(ticker)
    features = {
        'trend': analyze_trend(hist),
        'volume': analyze_volume(hist),
        'news_sentiment': analyze_news_sentiment(ticker),
        'earnings': has_upcoming_earnings(info),
        'sector_strength': 60,
        'insider': get_insider_score(ticker),
        'peg': get_peg_score(info),
        'fcf': get_fcf_score(info),
        'roic': get_roic_score(info)
    }
    score = compute_total_score(features)
    rec = recommend_strategy(score, info)
    trade, potential = suggest_trade_details(ticker, hist, rec)
    return {
        'ticker': ticker,
        'score': score,
        'recommendation': rec,
        'trade': trade,
        'potential': potential
    }

if __name__ == '__main__':
    recommendations = []
    for t in TICKER_LIST:
        try:
            print(f"Analyzing {t}...")
            result = run_analysis(t)
            if result['score'] >= THRESHOLD_SCORE:
                recommendations.append(result)
        except Exception as e:
            print(f"Error with {t}: {e}")

    if not recommendations:
        print("\nNo trades met the threshold score.")
    else:
        for rec in recommendations:
            print("\n---")
            print(f"Ticker: {rec['ticker']}")
            print(f"Score: {rec['score']:.2f}")
            print(f"Strategy: {rec['recommendation']}")
            print(f"Suggested Trade: {rec['trade']}")
            print(f"Expected Outcome: {rec['potential']}")
