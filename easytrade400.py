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

# Global cache to avoid repeated network calls
quiver_cache = {}

def robust_request_with_cache(url, headers, retries=3, delay=2):
    if url in quiver_cache:
        return quiver_cache[url]

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                quiver_cache[url] = response
                return response
            else:
                print(f"Non-200 response ({response.status_code}) for {url}")
        except requests.exceptions.Timeout:
            print(f"Timeout fetching {url}. Retrying ({attempt + 1}/{retries})...")
            time.sleep(delay * (attempt + 1))
        except Exception as e:
            print(f"Request error for {url}: {e}")
            break
    return None

def fetch_all_quiver_data():
    headers = {
        "Authorization": f"Bearer {QUIVER_API_KEY}",
        "Accept": "application/json"
    }
    corp_url = "https://api.quiverquant.com/beta/live/insiders"
    cong_url = "https://api.quiverquant.com/beta/live/congresstrading"

    corp_resp = robust_request_with_cache(corp_url, headers)
    cong_resp = robust_request_with_cache(cong_url, headers)

    corp_data = corp_resp.json() if corp_resp else []
    cong_data = cong_resp.json() if cong_resp else []

    return corp_data, cong_data

# Initialize API clients
analyzer = SentimentIntensityAnalyzer()

stocks = {
    'Top_Tier': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
    'Mid_Tier': [
        'SOFI', 'NXT', 'MGNI', 'JOBY', 'SKYE', 'F',
        'DKNG', 'U', 'CHPT', 'RUN', 'IONQ', 'PLUG', 'UPST', 'BMBL', 'PINS', 'ROKU', 'W', 'ETSY', 'PTON'
    ],
    'Political': ['BA', 'LMT', 'RTX', 'NOC', 'NANC'],
    'EV': ['TSLA', 'LCID', 'RIVN', 'NIO', 'FISK', 'XPEV'],
    'Energy': ['XOM', 'CVX', 'SLB', 'BP', 'MPC', 'HES'],
    'Tech': [
        'AMD', 'PLTR', 'INTC', 'CRM', 'ORCL',
        'AI', 'PATH', 'SNOW', 'ZS', 'NET', 'DDOG', 'MDB', 'CRWD'
    ],
    'High_Risk': [
        'GME', 'AMC', 'BBBY', 'MARA', 'RIOT', 'HOOD', 'COIN', 'TQQQ', 'SPXL', 'ARKK'
    ]
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

import time

def robust_request(url, headers, retries=3, delay=2):
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                return response
            else:
                print(f"Non-200 response ({response.status_code}) for {url}")
        except requests.exceptions.Timeout:
            print(f"Timeout fetching {url}. Retrying ({attempt + 1}/{retries})...")
            time.sleep(delay * (attempt + 1))
        except Exception as e:
            print(f"Request error for {url}: {e}")
            break
    return None

def fetch_insider_trading(ticker, corp_data, congress_data):
    try:
        if not isinstance(ticker, str):
            return 0

        score = 0

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
        print(f"Error scoring insider data for {ticker}: {e}")
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

def calculate_composite_score(ticker, corp_data, congress_data):
    stock_data = fetch_stock_data(ticker)
    if not stock_data:
        return None

    news_score = fetch_news_sentiment(ticker)
    insider_score = fetch_insider_trading(ticker, corp_data, congress_data)
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
    
def determine_trade_strategy(score, options_data):
    composite = score['score']
    news_score = score['news_score']
    tech_score = score['tech_score']
    insider_score = score['insider_score']
    political_score = score['political_score']
    social_score = score['social_score']

    # Filter for liquidity (optional advanced checks can go here)
    def is_liquid(option):
        spread = option['ask'] - option['bid']
        if spread <= 0 or option['ask'] == 0:
            return False
        bid_ask_pct = spread / option['ask']
        return bid_ask_pct <= 0.15 and option.get('openInterest', 100) >= 100

    if not all([
        is_liquid(options_data['put']),
        is_liquid(options_data['call_buy']),
        is_liquid(options_data['call_sell'])
    ]):
        return None

    # Strategy logic
    if composite >= 0.6:
        if tech_score > 0.2:
            option_type = 'Bullish Put (Sell CSP)'
            sell = options_data['put']
            limit_price = round(sell['bid'] * 0.95, 2)
            strike = sell['strike']
        else:
            return None
    elif 0.35 <= composite < 0.6:
        option_type = 'Bull Call Spread'
        buy = options_data['call_buy']
        sell = options_data['call_sell']
        spread_cost = buy['ask'] - sell['bid']
        limit_price = round(spread_cost * 1.05, 2)
        strike = f"{buy['strike']} / {sell['strike']}"
    elif 0.3 <= composite < 0.35:
        option_type = 'Bullish Call'
        buy = options_data['call_buy']
        limit_price = round(buy['ask'] * 1.05, 2)
        strike = buy['strike']
    elif composite <= -0.4:
        option_type = 'Bearish Put (Buy Put)'
        buy = options_data['put']
        limit_price = round(buy['ask'] * 1.05, 2)
        strike = buy['strike']
    else:
        return None

    # Tiering logic
    tier = "A+" if composite >= 0.75 else "A" if composite >= 0.6 else "B" if composite >= 0.35 else "Watch"
    confidence = round(min(composite * 100, 95), 2)
    if news_score < 0 or tech_score <= 0:
        confidence = min(confidence, 60)

    explanation = (
        f"{score['ticker']} is showing strong signals with a composite score of {round(composite, 2)}. "
        f"Tier: {tier}. "
        f"Technicals indicate {'oversold' if tech_score > 0.2 else 'neutral' if tech_score > 0 else 'overbought'} momentum. "
        f"News sentiment is {'positive' if news_score > 0 else 'negative'}, "
        f"and social media discussions are {'favorable' if social_score > 0 else 'unfavorable'}. "
        f"{'Notable political interest' if political_score > 0 else 'Low political exposure'}. "
        f"{'Recent insider purchases detected.' if insider_score > 0 else 'No recent insider support.'}"
    )

    return {
        'ticker': score['ticker'],
        'option_type': option_type,
        'expiration': options_data['expiration'],
        'strike': strike,
        'limit_price': limit_price,
        'confidence': confidence,
        'explanation': explanation,
        'tier': tier
    }


def generate_trade_recommendations(corp_data, congress_data):
    group_weights = {
        'Top_Tier': 1.0,
        'Mid_Tier': 1.1,
        'Political': 1.05,
        'EV': 1.2,
        'Energy': 1.0,
        'Tech': 1.0
    }

    scores = []
    for ticker in stock_list:
        score_data = calculate_composite_score(ticker, corp_data, congress_data)
        if score_data:
            group = next((g for g, lst in stocks.items() if ticker in lst), None)
            multiplier = group_weights.get(group, 1.0)
            score_data['score'] *= multiplier
            scores.append(score_data)

    scores = sorted(scores, key=lambda x: x['score'], reverse=True)

    trades = []
    seen_sectors = {}
    for score in scores:
        ticker = score['ticker']
        sector = next((g for g, lst in stocks.items() if ticker in lst), 'Other')
        if seen_sectors.get(sector, 0) >= 1:
            continue

        options_data = fetch_options_data(ticker)
        if not options_data:
            continue

        trade = determine_trade_strategy(score, options_data)
        if trade:
            trades.append(trade)
            seen_sectors[sector] = seen_sectors.get(sector, 0) + 1
        if len(trades) >= 5:
            break

    return trades


def main():
    print("Generating Options Trade Recommendations...")

    # Fetch QuiverQuant data once
    corp_data, congress_data = fetch_all_quiver_data()

    # Generate top recommendations using shared data
    trades = generate_trade_recommendations(corp_data, congress_data)

    # Save results
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

    # Show lowest scoring ticker
    all_scores = []
    for t in stock_list:
        try:
            s = calculate_composite_score(t, corp_data, congress_data)
            if s:
                all_scores.append(s)
        except Exception as e:
            print(f"Error scoring {t}: {e}")
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
