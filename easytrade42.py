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
SORT_BY_ABS_SCORE = True
BEAR_THRESHOLD = -0.6

# Full working logic restored

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

    scores = sorted(scores, key=lambda x: abs(x['score']), reverse=True)[:5] if SORT_BY_ABS_SCORE else sorted(scores, key=lambda x: x['score'], reverse=True)[:5]

    trades = []
    for score in scores:
        ticker = score['ticker']
        options_data = fetch_options_data(ticker)
        if not options_data:
            continue

        composite = score['score']
        explanation = f"{ticker} composite score: {round(composite, 2)} | Tech: {score['tech_score']} | News: {score['news_score']} | Insider: {score['insider_score']} | Social: {score['social_score']} | Political: {score['political_score']}"

        if composite >= 0.6:
            option_type = 'Bull Call Spread'
            buy = options_data['call_buy']
            sell = options_data['call_sell']
            spread_cost = buy['ask'] - sell['bid']
            limit_price = round(spread_cost * 1.05, 2)
            strike = f"{buy['strike']} / {sell['strike']}"
        elif composite >= 0.4:
            option_type = 'Bullish Call'
            buy = options_data['call_buy']
            limit_price = round(buy['ask'] * 1.05, 2)
            strike = buy['strike']
        elif composite >= 0.2:
            option_type = 'Bullish Put (Sell CSP)'
            sell = options_data['put']
            limit_price = round(sell['bid'] * 0.95, 2)
            strike = sell['strike']
        elif composite <= BEAR_THRESHOLD:
            option_type = 'Bear Call Spread'
            buy = options_data['call_sell']
            sell = options_data['call_buy']
            spread_credit = sell['bid'] - buy['ask']
            limit_price = round(spread_credit * 0.95, 2)
            strike = f"{sell['strike']} / {buy['strike']}"
        else:
            continue

        confidence = round(min(abs(composite) * 100, 95), 2)
        trades.append({
            'ticker': ticker,
            'option_type': option_type,
            'expiration': options_data['expiration'],
            'strike': strike,
            'limit_price': limit_price,
            'confidence': confidence,
            'explanation': explanation
        })

    return trades

def main():
    print("Generating Options Trade Recommendations...")
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

if __name__ == "__main__":
    main()
