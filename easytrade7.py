import yfinance as yf
import finnhub
from newsapi import NewsApiClient
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# API Keys (replace with your own)
FINNHUB_API_KEY = 'd1ijut1r01qhbuvqfv60d1ijut1r01qhbuvqfv6g'
NEWSAPI_KEY = '752ce0d986ec40d09652896b2315613a'

# Initialize API clients
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
newsapi = NewsApiClient(api_key=NEWSAPI_KEY)
analyzer = SentimentIntensityAnalyzer()

# Define stock universe (top-tier, mid-tier, political, EV, energy, tech)
stocks = {
    'Top_Tier': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
    'Mid_Tier': ['SOFI', 'NXT', 'MGNI', 'JOBY', 'SKYE'],
    'Political': ['BA', 'LMT', 'RTX', 'NOC'],  # Defense stocks tied to political events
    'EV': ['TSLA', 'LCID', 'RIVN', 'NIO'],
    'Energy': ['XOM', 'CVX', 'SLB', 'BP'],
    'Tech': ['AMD', 'PLTR', 'INTC', 'CRM', 'ORCL']
}

# Flatten stock list for analysis
stock_list = [stock for sector in stocks.values() for stock in sector]

def fetch_stock_data(ticker):
    """Fetch stock price, volume, and technical indicators."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='1mo', interval='1d')
        if hist.empty:
            return None
        
        # Calculate technical indicators
        rsi = calculate_rsi(hist['Close'])
        macd, signal = calculate_macd(hist['Close'])
        sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
        current_price = hist['Close'].iloc[-1]
        
        # Score based on technicals (e.g., RSI < 30 bullish, MACD crossover)
        tech_score = 0
        if rsi[-1] < 30:
            tech_score += 0.3  # Oversold
        elif rsi[-1] > 70:
            tech_score -= 0.3  # Overbought
        if macd[-1] > signal[-1] and macd[-2] <= signal[-2]:
            tech_score += 0.4  # Bullish MACD crossover
        if current_price > sma_20:
            tech_score += 0.2  # Above 20-day SMA
        
        return {
            'price': current_price,
            'volume': hist['Volume'].iloc[-1],
            'rsi': rsi[-1],
            'macd': macd[-1],
            'tech_score': tech_score
        }
    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {e}")
        return None

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index."""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
    rsi = [100 - (100 / (1 + rs))]
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
        rsi.append(100 - (100 / (1 + rs)))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD and signal line."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def fetch_news_sentiment(ticker):
    try:
        query = f"{ticker} stock"
        articles = newsapi.get_everything(q=query, language='en', sort_by='relevancy', page_size=20)
        if 'status' in articles and articles['status'] == 'error':
            print(f"Rate limit or error fetching news for {ticker}")
            return 0
        sentiment_score = 0
        count = 0
        for article in articles['articles']:
            title = article.get('title') or ''
            desc = article.get('description') or ''
            text = (title + ' ' + desc).lower()
            sentiment = analyzer.polarity_scores(text)
            sentiment_score += sentiment['compound']
            count += 1
        return sentiment_score / count if count > 0 else 0
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return 0

def scrape_x_sentiment(ticker):
    """Scrape sentiment from X posts (simplified)."""
    try:
        url = f"https://x.com/search?q={ticker}%20stock&src=typed_query"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        posts = soup.find_all('div', {'data-testid': 'tweetText'}, limit=10)
        sentiment_score = 0
        count = 0
        for post in posts:
            text = post.get_text().lower()
            sentiment = analyzer.polarity_scores(text)
            sentiment_score += sentiment['compound']
            count += 1
        return sentiment_score / count if count > 0 else 0
    except Exception as e:
        print(f"Error scraping X for {ticker}: {e}")
        return 0

def fetch_insider_trading(ticker):
    """Fetch insider trading data from Finnhub."""
    try:
        insiders = finnhub_client.stock_insider_transactions(ticker, _from='2025-01-01', to='2025-07-01')
        insider_score = 0
        if 'data' not in insiders:
            return 0
        for trade in insiders['data'][:10]:
            if trade.get('transactionType', '').lower() == 'buy':
                insider_score += 0.2
            elif trade.get('transactionType', '').lower() == 'sell':
                insider_score -= 0.2
        return insider_score
    except Exception as e:
        print(f"Error fetching insider data for {ticker}: {e}")
        return 0

def fetch_options_data(ticker):
    """Fetch options data and select optimal option."""
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return None
        # Select an expiration ~30-45 days out
        target_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
        expiration = min(expirations, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.strptime(target_date, '%Y-%m-%d')).days))
        
        calls = stock.option_chain(expiration).calls
        if calls.empty:
            return None
        
        # Select ATM or slightly OTM call
        current_price = stock.history(period='1d')['Close'].iloc[-1]
        calls['moneyness'] = abs(calls['strike'] - current_price)
        atm_call = calls.loc[calls['moneyness'].idxmin()]
        
        return {
            'expiration': expiration,
            'strike': atm_call['strike'],
            'last_price': atm_call['lastPrice'],
            'bid': atm_call['bid'],
            'ask': atm_call['ask'],
            'implied_volatility': atm_call['impliedVolatility']
        }
    except Exception as e:
        print(f"Error fetching options for {ticker}: {e}")
        return None

def calculate_composite_score(ticker):
    """Calculate composite score based on multiple factors."""
    stock_data = fetch_stock_data(ticker)
    if not stock_data:
        return None
    
    news_score = fetch_news_sentiment(ticker)
    x_score = scrape_x_sentiment(ticker)
    insider_score = fetch_insider_trading(ticker)
    
    # Sector-specific political factors (simplified)
    sector = next((k for k, v in stocks.items() if ticker in v), 'Other')
    political_score = 0
    if sector == 'EV':
        political_score += 0.2 if 'subsidy' in fetch_news_sentiment(ticker.lower()) else 0
    elif sector == 'Energy':
        political_score += 0.2 if 'regulation' in fetch_news_sentiment(ticker.lower()) else 0
    elif sector == 'Political':
        political_score += 0.3 if 'defense' in fetch_news_sentiment(ticker.lower()) else 0
    
    # Composite score (weighted)
    composite_score = (
        stock_data['tech_score'] * 0.4 +
        news_score * 0.2 +
        x_score * 0.2 +
        insider_score * 0.1 +
        political_score * 0.1
    )
    
    return {
        'ticker': ticker,
        'score': composite_score,
        'tech_score': stock_data['tech_score'],
        'news_score': news_score,
        'x_score': x_score,
        'insider_score': insider_score,
        'political_score': political_score,
        'price': stock_data['price']
    }

def generate_trade_recommendations():
    """Generate top 3 options trade recommendations."""
    scores = []
    for ticker in stock_list:
        score_data = calculate_composite_score(ticker)
        if score_data:
            scores.append(score_data)
    
    # Sort by composite score
    scores = sorted(scores, key=lambda x: x['score'], reverse=True)[:3]
    
    trades = []
    for score in scores:
        ticker = score['ticker']
        options_data = fetch_options_data(ticker)
        if not options_data:
            continue
        
        # Determine option type (bullish call for simplicity)
        option_type = 'Bullish Call'
        limit_price = options_data['ask'] * 1.05  # Add 5% buffer
        confidence = min(score['score'] * 100, 95)  # Cap at 95%
        
        # Explanation
        explanation = f"{ticker} is recommended due to: "
        if score['tech_score'] > 0.5:
            explanation += f"Strong technical indicators (RSI: {score['rsi']:.2f}, MACD crossover); "
        if score['news_score'] > 0.3:
            explanation += "Positive news sentiment; "
        if score['x_score'] > 0.3:
            explanation += "Bullish sentiment on X; "
        if score['insider_score'] > 0:
            explanation += "Recent insider buying; "
        if score['political_score'] > 0:
            explanation += "Favorable political developments; "
        
        trades.append({
            'ticker': ticker,
            'option_type': option_type,
            'expiration': options_data['expiration'],
            'strike': options_data['strike'],
            'limit_price': round(limit_price, 2),
            'confidence': round(confidence, 2),
            'explanation': explanation
        })
    
    return trades

def main():
    print("Generating Options Trade Recommendations...")
    trades = generate_trade_recommendations()
    
    print("\nTop 3 Options Trades:")
    for i, trade in enumerate(trades, 1):
        print(f"\nTrade {i}:")
        print(f"Ticker: {trade['ticker']}")
        print(f"Option Type: {trade['option_type']}")
        print(f"Expiration Date: {trade['expiration']}")
        print(f"Strike Price: ${trade['strike']}")
        print(f"Limit Price: ${trade['limit_price']}")
        print(f"Confidence Level: {trade['confidence']}%")
        print(f"Explanation: {trade['explanation']}")
    
    if not trades:
        print("No suitable trades found. Try adjusting parameters or check API connectivity.")

if __name__ == "__main__":
    main()
