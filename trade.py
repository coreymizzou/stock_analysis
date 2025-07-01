https://github.com/coreymizzou/stock_analysis.gi# personal_trader_bot.py

"""
Modular Python script to act as a personal investor/trader.
Goals:
- Maximize income from stocks and options trading
- Use real-time data and insider activity to drive decisions
"""

# =====================
# Imports and Constants
# =====================
import yfinance as yf
import pandas as pd
import requests
import datetime
import json
import logging
from typing import List, Dict, Tuple

# Logging configuration
logging.basicConfig(filename='trade_log.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Alpaca keys (optional for live/paper trading)
ALPACA_API_KEY = 'your_alpaca_api_key'
ALPACA_SECRET_KEY = 'your_alpaca_secret_key'
BASE_URL = 'https://paper-api.alpaca.markets'

# =====================
# 1. Data Collection
# =====================
def fetch_stock_data(ticker: str) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    hist = stock.history(period="3mo")
    return hist

def fetch_insider_data(ticker: str) -> List[Dict]:
    url = f"https://www.openinsider.com/screener?s={ticker}&o=&pl=&ph=&ll=&lh=&fd=0&fdr=&td=0&tdr=&xp=1&vl=&vh=&ocl=&och=&sic1=&sic2=&sortcol=0&maxresults=10"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        tables = pd.read_html(response.text)
        if not tables:
            return []
        df = tables[0]
        records = []
        for _, row in df.iterrows():
            try:
                records.append({
                    "date": row["Trade Date"],
                    "insider": row["Insider Name"],
                    "type": row["Transaction Type"],
                    "shares": int(str(row["Shares Traded"]).replace(',', '')),
                    "price": float(str(row["Price"]).replace('$', '').replace(',', ''))
                })
            except Exception:
                continue
        return records
    except Exception as e:
        logging.warning(f"Insider data fetch failed for {ticker}: {e}. Using fallback mock data.")
        return [{
            "date": str(datetime.date.today()),
            "insider": "Mock Insider",
            "type": "Buy",
            "shares": 15000,
            "price": 75.50
        }]
        df = tables[0]
        records = []
        for _, row in df.iterrows():
            try:
                records.append({
                    "date": row["Trade Date"],
                    "insider": row["Insider Name"],
                    "type": row["Transaction Type"],
                    "shares": int(str(row["Shares Traded"]).replace(',', '')),
                    "price": float(str(row["Price"]).replace('$', '').replace(',', ''))
                })
            except Exception:
                continue
        return records
    except Exception as e:
        logging.warning(f"Insider data fetch failed for {ticker}: {e}")
        return []

def fetch_option_chain(ticker: str) -> Dict:
    stock = yf.Ticker(ticker)
    expirations = stock.options
    if not expirations:
        return {}
    try:
        options = stock.option_chain(expirations[0])
        calls = options.calls.copy()
        puts = options.puts.copy()

        calls['spread'] = calls['ask'] - calls['bid']
        puts['spread'] = puts['ask'] - puts['bid']

        filtered_calls = calls[(calls["delta"] >= 0.4) & (calls["delta"] <= 0.6) &
                               (calls["openInterest"] > 500) & (calls['spread'] < 0.5)]
        filtered_puts = puts[(puts["delta"] >= -0.6) & (puts["delta"] <= -0.4) &
                             (puts["openInterest"] > 500) & (puts['spread'] < 0.5)]

        filtered_calls['pop'] = 1 - (filtered_calls['impliedVolatility'] * 0.4)
        filtered_puts['pop'] = 1 - (filtered_puts['impliedVolatility'] * 0.4)

        return {
            "calls": filtered_calls.to_dict("records"),
            "puts": filtered_puts.to_dict("records"),
            "expiration": expirations[0]
        }
    except Exception as e:
        logging.warning(f"Failed to fetch option chain for {ticker}: {e}")
        return {}

# =====================
# 2. Ticker Universe
# =====================
TOP_TIER = [
    "SPY", "QQQ", "DIA",
    "AAPL", "AMZN", "MSFT", "NVDA", "GOOGL", "META", "TSLA",
    "BRK.B", "V", "UNH", "JPM", "MA", "AVGO", "NFLX"
]

MID_TIER = [
    "PLTR", "SOFI", "DKNG", "FUBO", "RBLX", "RUN", "BLNK",
    "CHPT", "OPEN", "LCID", "UPST", "AFRM", "COIN", "HOOD", "TTD"
]

UPCOMING = [
    "IONQ", "ASTS", "GCT", "VLCN", "HLLY", "AMRS", "NNDM", "INM", "BMEA",
    "HIMS", "BIRD", "ENVX", "SOUN", "PRST", "BEEM", "CANO", "PRZO"
]

SECTORS = {
    "EV": ["TSLA", "NIO", "RIVN", "LCID", "XPEV", "F", "GM", "LI", "NKLA"],
    "Energy": ["XOM", "CVX", "SLB", "ENPH", "FSLR", "NEE", "OXY", "COP", "VLO"],
    "Tech/AI": ["NVDA", "AMD", "PLTR", "AI", "SMCI", "CRWD", "ZS", "DDOG", "SNOW", "MDB"],
    "Defense": ["LMT", "RTX", "NOC", "BA", "PLTR", "CRWD", "PANW"],
    "Biotech": ["MRNA", "VRTX", "BIIB", "NVAX", "REGN", "BNTX", "SRPT"],
    "Politics": ["NANC"]
}

# =====================
# 3. Strategy Engine
# =====================
def analyze_technical_indicators(df: pd.DataFrame) -> Dict:
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    latest = df.iloc[-1]
    return {
        "price": latest["Close"],
        "sma20": latest["SMA_20"],
        "sma50": latest["SMA_50"],
        "bullish": latest["SMA_20"] > latest["SMA_50"]
    }

def recommend_option_trade(ticker: str, signal: Dict, insider_data: List[Dict], option_chain: Dict) -> Dict:
    if not option_chain:
        return {"type": "Watch", "confidence": "No Liquidity"}

    call = option_chain['calls'][0] if option_chain['calls'] else {}
    put = option_chain['puts'][0] if option_chain['puts'] else {}

    if signal['bullish'] and any(d['type'] == 'Buy' for d in insider_data):
        return {
            "type": "Buy Call",
            "strike": call.get("strike"),
            "expiration": option_chain['expiration'],
            "delta": call.get("delta"),
            "open_interest": call.get("openInterest"),
            "pop": call.get("pop"),
            "confidence": "Strong Buy"
        }
    elif signal['bullish'] and not insider_data:
        return {
            "type": "Sell Put",
            "strike": put.get("strike"),
            "expiration": option_chain['expiration'],
            "delta": put.get("delta"),
            "open_interest": put.get("openInterest"),
            "pop": put.get("pop"),
            "confidence": "Income Trade"
        }
    elif not signal['bullish'] and any(d['type'] == 'Sell' for d in insider_data):
        return {
            "type": "Buy Put",
            "strike": put.get("strike"),
            "expiration": option_chain['expiration'],
            "delta": put.get("delta"),
            "open_interest": put.get("openInterest"),
            "pop": put.get("pop"),
            "confidence": "Speculative"
        }
    else:
        return {"type": "Watch", "confidence": "Watchlist"}

# =====================
# 4. Outputs
# =====================
def generate_report(ticker: str, sector: str, insider: List[Dict], techs: Dict, option: Dict):
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, (str, int, float, type(None))):
            return obj
        else:
            return str(obj)

    report = {
        "Ticker": ticker,
        "Sector": sector,
        "Insider Activity": insider,
        "Technical Signals": techs,
        "Option Trade Recommendation": option
    }
    filename = f"report_{ticker}_{datetime.date.today()}.json"
    with open(filename, 'w') as f:
        json.dump(clean_for_json(report), f, indent=2)
    print(f"{ticker}: {option['type']} | Strike: {option.get('strike')} | Confidence: {option['confidence']}")
    print(f"{ticker}: {option['type']} | Strike: {option.get('strike')} | Confidence: {option['confidence']}")

# =====================
# 5. Run Analysis
# =====================
def run_daily_analysis():
    tickers = TOP_TIER + MID_TIER + UPCOMING + sum(SECTORS.values(), [])
    seen = set()
    for ticker in tickers:
        if ticker in seen:
            continue
        seen.add(ticker)
        try:
            sector = next((s for s, lst in SECTORS.items() if ticker in lst), "Uncategorized")
            df = fetch_stock_data(ticker)
            insider = fetch_insider_data(ticker)
            techs = analyze_technical_indicators(df)
            option_chain = fetch_option_chain(ticker)
            option = recommend_option_trade(ticker, techs, insider, option_chain)
            generate_report(ticker, sector, insider, techs, option)
            logging.info(f"Analyzed {ticker} successfully.")
        except Exception as e:
            logging.error(f"Error analyzing {ticker}: {e}")
            print(f"Error analyzing {ticker}: {e}")

if __name__ == "__main__":
    run_daily_analysis()
