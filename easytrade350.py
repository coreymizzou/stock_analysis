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
QUIVER_API_KEY = 'e1c45cb296aab1338edbef3e11fb9b2acd66413b'

# Initialize API clients
analyzer = SentimentIntensityAnalyzer()

stocks = {
    'Top_Tier': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
    'Mid_Tier': ['SOFI', 'NXT', 'MGNI', 'JOBY', 'SKYE', 'F'],
    'Political': ['BA', 'LMT', 'RTX', 'NOC', 'NANC'],
    'EV': ['TSLA', 'LCID', 'RIVN', 'NIO'],
    'Energy': ['XOM', 'CVX', 'SLB', 'BP'],
    'Tech': ['AMD', 'PLTR', 'INTC', 'CRM', 'ORCL']
}

stock_list = [stock for sector in stocks.values() for stock in sector]
SORT_BY_ABS_SCORE = True
BEAR_THRESHOLD = -0.6

def sector_limiter(trades, max_per_sector=2):
    sector_counts = {}
    filtered = []
    for trade in trades:
        sector = next((s for s, tickers in stocks.items() if trade['ticker'] in tickers), 'Other')
        if sector_counts.get(sector, 0) < max_per_sector:
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            filtered.append(trade)
    return filtered

def normalize_scores(scores):
    df = pd.DataFrame(scores)
    for key in ['tech_score', 'news_score', 'insider_score', 'social_score']:
        if df[key].std() == 0:
            df[key] = 0  # All values are same; no contribution to score
        else:
            df[key] = (df[key] - df[key].mean()) / df[key].std()
    
    df['normalized_score'] = (
        df['tech_score'] * 0.35 +
        df['news_score'] * 0.2 +
        df['insider_score'] * 0.15 +
        df['social_score'] * 0.2 +
        df['political_score'] * 0.1
    )

    df = df.replace([np.inf, -np.inf], 0).dropna()
    return df.sort_values('normalized_score', ascending=False).to_dict(orient='records')

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
    try:
        url = f"https://www.reddit.com/search.json?q={ticker}&limit=10"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return 0

        data = response.json()
        sentiment_score = 0
        count = 0

        for post in data.get("data", {}).get("children", []):
            title = post.get("data", {}).get("title", "").lower()
            sentiment = analyzer.polarity_scores(title)
            sentiment_score += sentiment['compound']
            count += 1

        return sentiment_score / count if count > 0 else 0
    except Exception as e:
        print(f"Error fetching social sentiment for {ticker}: {e}")
        return 0

def fetch_insider_trading(ticker):
    try:
        if not isinstance(ticker, str):
            return 0

        headers = {"Authorization": f"Bearer {QUIVER_API_KEY}", "Accept": "application/json"}
        congress_url = "https://api.quiverquant.com/beta/live/congresstrading"
        corp_url = "https://api.quiverquant.com/beta/live/insiders"

        congress_resp = requests.get(congress_url, headers=headers, timeout=10)
        corp_resp = requests.get(corp_url, headers=headers, timeout=10)

        score = 0

        if congress_resp.status_code == 200:
            congress_data = congress_resp.json()
            recent_congress = [t for t in congress_data if t.get("Ticker", "") and t["Ticker"].upper() == ticker.upper()][:10]
            for trade in recent_congress:
                action = trade.get("Transaction", "").lower()
                amount = float(trade.get("Amount", 0) or 0)
                rep_name = trade.get("Representative", "").lower()
                if "purchase" in action:
                    score += 0.2 + min(amount / 100000, 0.3)
                elif "sale" in action:
                    score -= 0.2 + min(amount / 100000, 0.3)
                if "pelosi" in rep_name:
                    score += 0.3

        if corp_resp.status_code == 200:
            corp_data = corp_resp.json()
            recent_corp = [t for t in corp_data if t.get("Ticker", "") and t["Ticker"].upper() == ticker.upper()][:10]
            for trade in recent_corp:
                action = trade.get("Transaction", "").lower()
                value = float(trade.get("Value", 0) or 0)
                if "purchase" in action:
                    score += 0.2 + min(value / 100000, 0.3)
                elif "sale" in action:
                    score -= 0.2 + min(value / 100000, 0.3)

        return score
    except Exception as e:
        print(f"Error fetching insider data for {ticker}: {e}")
        return 0

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

def calculate_composite_score(ticker):
    stock_data = fetch_stock_data(ticker)
    if not stock_data:
        return None

    news_score = fetch_news_sentiment(ticker)
    insider_score = fetch_insider_trading(ticker)
    social_score = fetch_social_sentiment(ticker)

    sector = next((k for k, v in stocks.items() if ticker in v), 'Other')
    political_score = 0.2 if sector in ['EV', 'Energy'] else 0.3 if sector == 'Political' else 0

    composite_score = (
        stock_data['tech_score'] * 0.35 +
        news_score * 0.2 +
        insider_score * 0.15 +
        political_score * 0.1 +
        social_score * 0.2
    )

    return {
        'ticker': ticker,
        'score': composite_score,
        'tech_score': stock_data['tech_score'],
        'news_score': news_score,
        'insider_score': insider_score,
        'political_score': political_score,
        'social_score': social_score,
        'price': stock_data['price']
    }

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
        calls = calls[calls['ask'] > 0]
        if calls.shape[0] < 2:
            return None
        call_buy = calls.iloc[0]
        call_sell = calls.iloc[1]

        puts['moneyness'] = abs(puts['strike'] - current_price)
        puts = puts[puts['bid'] > 0]
        if puts.empty:
            return None
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

    scores = sorted(scores, key=lambda x: x['score'], reverse=True)[:5]

    trades = []
    for score in scores:
        ticker = score['ticker']
        options_data = fetch_options_data(ticker)
        if not options_data:
            continue

        composite = score['score']
        if composite >= 0.6:
            option_type = 'Bullish Put (Sell CSP)'
            sell = options_data['put']
            limit_price = round(sell['bid'] * 0.95, 2)
            strike = sell['strike']
        elif composite >= 0.4:
            option_type = 'Bullish Call'
            buy = options_data['call_buy']
            limit_price = round(buy['ask'] * 1.05, 2)
            strike = buy['strike']
        elif composite >= 0.2:
            option_type = 'Bull Call Spread'
            buy = options_data['call_buy']
            sell = options_data['call_sell']
            spread_cost = buy['ask'] - sell['bid']
            limit_price = round(spread_cost * 1.05, 2)
            strike = f"{buy['strike']} / {sell['strike']}"
        elif composite <= -0.4:
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
            'explanation': f"{ticker} is showing strong signals with a composite score of {round(score['score'], 2)}. Technicals indicate {'oversold' if score['tech_score'] > 0.2 else 'neutral' if score['tech_score'] > 0 else 'overbought'} momentum. News sentiment is {'positive' if score['news_score'] > 0 else 'negative'}, and social media discussions are {'favorable' if score['social_score'] > 0 else 'unfavorable'}. {'Notable political interest' if score['political_score'] > 0 else 'Low political exposure'}. {'Recent insider purchases detected.' if score['insider_score'] > 0 else 'No recent insider support.'}"
        })

    return trades

def main():
    print("Generating Options Trade Recommendations...")
    from trade_logic import generate_trade_recommendations, calculate_composite_score, fetch_options_data

    trades = generate_trade_recommendations()

    pd.DataFrame(trades).to_csv("trade_recommendations.csv", index=False)
    print("Saved recommendations to trade_recommendations.csv")

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

    # Show lowest scored ticker
    all_scores = []
    for t in stock_list:
        try:
            s = calculate_composite_score(t)
            if s:
                all_scores.append(s)
        except:
            continue

    if all_scores:
        lowest = sorted(all_scores, key=lambda x: x['score'])[0]
        option_data = fetch_options_data(lowest['ticker'])
        if option_data:
            print("\nLowest Scoring Ticker:")
            print(f"\nTrade (Lowest Score):")
            print(f"Ticker: {lowest['ticker']}")
            print(f"Option Type: Bearish Put (Buy Put)")
            print(f"Expiration Date: {option_data['expiration']}")
            print(f"Strike: {option_data['put']['strike']}")
            print(f"Limit Price: ${round(option_data['put']['ask'] * 1.05, 2)}")
            print(f"Confidence Level: {round(min(abs(lowest['score']) * 100, 95), 2)}%")
            print(f"Explanation: {lowest['ticker']} is showing weak signals with a composite score of {round(lowest['score'], 2)}. Technicals suggest weakness. News and social sentiment are negative. Insider trading data shows recent sales or lack of purchases.")

if __name__ == "__main__":
    main()
