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

def calculate_composite_score(ticker):
    stock_data = fetch_stock_data(ticker)
    if not stock_data:
        return None

    news_score = fetch_news_sentiment(ticker)
    social_score = 0
    insider_score = fetch_insider_trading(ticker)

    sector = next((k for k, v in stocks.items() if ticker in v), 'Other')
    political_score = 0
    if sector == 'EV':
        political_score += 0.2
    elif sector == 'Energy':
        political_score += 0.2
    elif sector == 'Political':
        political_score += 0.3

    composite_score = (
        stock_data['tech_score'] * 0.3 +
        news_score * 0.2 +
        social_score * 0.1 +
        insider_score * 0.2 +
        political_score * 0.2
    )

    return {
        'ticker': ticker,
        'score': composite_score,
        'tech_score': stock_data['tech_score'],
        'news_score': news_score,
        'social_score': social_score,
        'insider_score': insider_score,
        'political_score': political_score,
        'price': stock_data['price']
    }

def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='1mo', interval='1d')
        if hist.empty:
            return None

        rsi = calculate_rsi(hist['Close'])
        macd, signal = calculate_macd(hist['Close'])
        sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
        current_price = hist['Close'].iloc[-1]

        tech_score = 0
        if rsi[-1] < 30:
            tech_score += 0.3
        elif rsi[-1] > 70:
            tech_score -= 0.3
        if macd[-1] > signal[-1] and macd[-2] <= signal[-2]:
            tech_score += 0.4
        if current_price > sma_20:
            tech_score += 0.2

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
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def fetch_insider_trading(ticker):
    try:
        url = f"https://openinsider.com/screener?s={ticker}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.text, 'html.parser')
        rows = soup.select('table.tinytable tr')[1:]  # skip header

        score = 0
        for row in rows[:10]:
            cols = row.find_all('td')
            if len(cols) > 0:
                trade_type = cols[6].text.strip().lower()
                if 'p' in trade_type:  # purchase
                    score += 0.2
                elif 's' in trade_type:  # sale
                    score -= 0.2
        return score
    except Exception as e:
        print(f"Error scraping insider data for {ticker}: {e}")
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
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return None

        target_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
        expiration = min(expirations, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - datetime.strptime(target_date, '%Y-%m-%d')).days))
        chain = stock.option_chain(expiration)

        calls = chain.calls
        puts = chain.puts

        current_price = stock.history(period='1d')['Close'].iloc[-1]

        calls['moneyness'] = calls['strike'] - current_price
        calls = calls[calls['moneyness'] >= 0].sort_values('moneyness')
        if calls.shape[0] < 2:
            return None

        call_buy = calls.iloc[0]
        call_sell = calls.iloc[1]
        puts['moneyness'] = abs(puts['strike'] - current_price)
        atm_put = puts.loc[puts['moneyness'].idxmin()]

        return {
            'expiration': expiration,
            'call_buy': call_buy,
            'call_sell': call_sell,
            'put': atm_put,
            'price': current_price
        }
    except Exception as e:
        print(f"Error fetching options for {ticker}: {e}")
        return None

def generate_trade_recommendations():
    scores = []
    for ticker in stock_list:
        score_data = calculate_composite_score(ticker)
        if score_data:
            scores.append(score_data)

    print("\nAll Composite Scores:")
    for s in sorted(scores, key=lambda x: x['score'], reverse=True):
        print(f"{s['ticker']}: score={round(s['score'], 2)} | tech={round(s['tech_score'], 2)} | news={round(s['news_score'], 2)} | insider={round(s['insider_score'], 2)} | political={round(s['political_score'], 2)}")

    scores = sorted(scores, key=lambda x: x['score'], reverse=True)[:5]

    trades = []
    for score in scores:
        ticker = score['ticker']
        options_data = fetch_options_data(ticker)
        if not options_data:
            continue

        composite = score['score']
        if composite >= 0.5:
            option_type = 'Bull Call Spread'
            buy = options_data['call_buy']
            sell = options_data['call_sell']
            spread_cost = buy['ask'] - sell['bid']
            limit_price = round(spread_cost * 1.05, 2)
            strike = f"{buy['strike']} / {sell['strike']}"
        elif composite >= 0.3:
            option_type = 'Bullish Call'
            buy = options_data['call_buy']
            limit_price = round(buy['ask'] * 1.05, 2)
            strike = buy['strike']
        elif composite >= 0.1:
            option_type = 'Bullish Put (Sell CSP)'
            sell = options_data['put']
            limit_price = round(sell['bid'] * 0.95, 2)
            strike = sell['strike']
        elif composite <= -0.3:
            option_type = 'Bearish Put (Buy Put)'
            buy = options_data['put']
            limit_price = round(buy['ask'] * 1.05, 2)
            strike = buy['strike']
        else:
            continue

        trades.append({
            'ticker': ticker,
            'option_type': option_type,
            'expiration': options_data['expiration'],
            'strike': strike,
            'limit_price': limit_price,
            'confidence': round(min(score['score'] * 100, 95), 2),
            'explanation': f"Composite score: {round(score['score'], 2)}"
        })

    return trades

def main():
    print("Generating Options Trade Recommendations...")
    trades = generate_trade_recommendations()

    print("\nTop Trade Recommendations:")
    for i, trade in enumerate(trades, 1):
        print(f"\nTrade {i}:")
        print(f"Ticker: {trade['ticker']}")
        print(f"Option Type: {trade['option_type']}")
        print(f"Expiration Date: {trade['expiration']}")
        print(f"Strike: {trade['strike']}")
        print(f"Limit Price: ${trade['limit_price']}")
        print(f"Confidence Level: {trade['confidence']}%")
        print(f"Explanation: {trade['explanation']}")

if __name__ == "__main__":
    main()
