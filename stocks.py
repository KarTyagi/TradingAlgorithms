import re
import requests
import yfinance as yf
from symbols import SYMBOLS

def get_stock_price_and_metrics(symbol):
    api_key = "SlOrGMWxbNwTiJGDfIrkswILZdxmv6O3"
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol.upper()}?timeseries=2&apikey={api_key}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'historical' in data and len(data['historical']) > 0:
                latest_bar = data['historical'][0]  # newest first
                price = latest_bar['close']
                volume = latest_bar['volume']
                # Approximate trades as volume (not accurate, but for calculation)
                trades = volume  # or '' 
                return price, volume, trades
        return '', '', ''
    except Exception as e:
        print(f"[DEBUG] FMP API error (current) for {symbol}: {e}")
        return '', '', ''

def extract_sp500_mentions(text):
    matches = set()
    for symbol, names in SYMBOLS.items():
        if re.search(rf'\b{re.escape(symbol)}\b', text, re.IGNORECASE):
            matches.add((symbol, names.split('|')[0]))
            continue
        for name in names.split('|'):
            if re.search(rf'\b{re.escape(name)}\b', text, re.IGNORECASE):
                matches.add((symbol, name))
                break
    return list(matches)

def format_volume(val):
    try:
        val = float(val)
        if val >= 1_000_000_000:
            return f"{val/1_000_000_000:.1f}b"
        elif val >= 1_000_000:
            return f"{val/1_000_000:.1f}m"
        elif val >= 1_000:
            return f"{val/1_000:.1f}k"
        else:
            return str(int(val))
    except Exception:
        return str(val)
