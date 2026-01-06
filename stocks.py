import re
import requests
from symbols import SYMBOLS

def get_stock_price_and_metrics(symbol):
    api_key = "2YzDBsof_v5re2giJiCzkB2l1pX8dTyn"
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol.upper()}/prev?adjusted=true&apiKey={api_key}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'results' in data and len(data['results']) > 0:
                result = data['results'][0]
                close = result.get('c', '')
                volume = result.get('v', '')
                num_trades = result.get('n', '')
                return close, volume, num_trades
        return '', '', ''
    except Exception as e:
        print(f"[DEBUG] Polygon API error for {symbol}: {e}")
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
