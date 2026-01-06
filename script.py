import os
import pandas as pd
import numpy as np
from tabulate import tabulate
from datetime import datetime
from sentiment import initialize_analyzer, initialize_absa, analyze_sentiment, analyze_aspect_sentiment
from news import scrape_yf, fetch_article_text
from stocks import get_stock_price_and_metrics, extract_sp500_mentions
import requests
from scipy.stats import norm


def get_fmp_historical_closes(symbol, days=200, api_key="SlOrGMWxbNwTiJGDfIrkswILZdxmv6O3"):
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol.upper()}?timeseries={days}&apikey={api_key}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'historical' in data:
                closes = [bar['close'] for bar in data['historical']]
                closes.reverse()
                return closes
        return []
    except Exception as e:
        print(f"[DEBUG] FMP API error (historical) for {symbol}: {e}")
        return []


def calculate_moving_averages(closes, windows=[20, 50, 200]):
    ma = {}
    for w in windows:
        if len(closes) >= w:
            ma[w] = round(sum(closes[-w:]) / w, 2)
        else:
            ma[w] = None
    return ma


def main():
    analyzer = initialize_analyzer()

    absa_pipeline = initialize_absa()
    
    target_date = datetime.now().date()
    
    print(f"Fetching articles from Yahoo Finance for {target_date}...")
    articles = scrape_yf(target_date)
    print(f"[DEBUG] scrape_yf returned {len(articles)} articles before filtering.")
    if len(articles) > 25:
        articles = articles[:25]
    print(f"[DEBUG] scrape_yf using {len(articles)} articles after slicing to 25.")
    if not articles:
        print("No articles available to analyze. Check site structure or date availability.")
        return
    if os.path.exists('sentiment_results.csv'):
        try:
            prev_df = pd.read_csv('sentiment_results.csv')
            if 'Attempt' in prev_df.columns:
                _ = prev_df['Attempt'].max() + 1
        except Exception:
            pass
    table_data = []
    closes_cache = {}
    ma_cache = {}
    for idx, article in enumerate(articles, 1):
        title = article["title"]
        print(f"[DEBUG] Article {idx}: {title}")
        mentions = extract_sp500_mentions(title)
        print(f"[DEBUG] Mentions found: {mentions}")
        if not mentions:
            continue
        article_text = fetch_article_text(article["link"])
        short_date = str(target_date)[2:]
        for symbol, company in mentions:
            print(f"[DEBUG] Matched symbol: {symbol}, company: {company}")
            stock_price, stock_volume, stock_trades = get_stock_price_and_metrics(symbol)
            try:
                vpt = float(stock_volume) / float(stock_trades) if float(stock_trades) > 0 else 0
            except Exception:
                vpt = 0
            vpt_str = f"{vpt:,.1f}" if vpt else ""
            context = article_text if article_text else title
            keywords_positive = ["upgrade", "buy", "raise", "outperform", "beat", "growth", "profit", "merge", "acquisition"]
            keywords_negative = ["layoff", "cut", "downgrade", "sell", "underperform", "miss", "loss", "decline", "lawsuit", "fail"]
            context_lower = context.lower()
            keyword_sentiment = None
            if any(word in context_lower for word in keywords_positive):
                keyword_sentiment = "positive"
            elif any(word in context_lower for word in keywords_negative):
                keyword_sentiment = "negative"
            try:
                label, raw_score, mapped_score = analyze_aspect_sentiment(absa_pipeline, context, symbol)
                if label == "neutral":
                    if keyword_sentiment:
                        label = keyword_sentiment
                        mapped_score = 1.0 if keyword_sentiment == "positive" else -1.0
                    else:
                        reg_result = analyze_sentiment(analyzer, context)
                        reg_label = reg_result.get("label", "neutral").lower()
                        if reg_label in ("positive", "negative"):
                            label = reg_label
                            mapped_score = 1.0 if reg_label == "positive" else -1.0
                print(f"[DEBUG] Sentiment for {symbol}: label={label}, raw_score={raw_score}, mapped_score={mapped_score}")
                mapped_score_val = round(mapped_score, 3) if isinstance(mapped_score, (int, float)) else ""
            except Exception as e:
                print(f"[DEBUG] Sentiment analysis error for {symbol}: {e}")
            table_data.append([
                idx,
                short_date,
                symbol,
                vpt_str,
                company,
                (title[:30] + "...") if len(title) > 33 else title,
                mapped_score_val,
                stock_price,
                stock_volume,
                stock_trades
            ])
    headers = ["#", "Date", "Symbol", "Company Name", "Article Title", "Mapped Score", "s_8"]
    print_headers = headers
    print_table_data = []
    phi = None
    t = None
    s_8 = None
    s_8_list = []
    for i, row in enumerate(table_data):
        symbol = row[2]
        title = row[5] if len(row) > 5 and row[5] not in (None, '', 'N/A') else ''
        s_6 = ''
        price = row[7] if len(row) > 7 and row[7] not in (None, '', 'N/A') else None
        mapped_score_val = row[6] if len(row) > 6 and isinstance(row[6], (int, float, str)) else ''
        vpt_str = row[3]
        valid_row = bool(symbol and price not in (None, '', 'N/A'))
        vpt = None
        try:
            vpt = float(vpt_str.replace(',', '')) if isinstance(vpt_str, str) and vpt_str else float(vpt_str)
        except Exception:
            vpt = None
        if valid_row:
            if symbol not in closes_cache:
                closes_cache[symbol] = get_fmp_historical_closes(symbol, days=200)
            closes = closes_cache[symbol]
            if symbol not in ma_cache:
                ma_cache[symbol] = calculate_moving_averages(closes)
            ma = ma_cache[symbol]
            m20, m50, m200 = ma[20], ma[50], ma[200]
            mf = ''
            if all(isinstance(x, (int, float)) and x not in (None, 0) for x in [m20, m50, m200]):
                try:
                    mf = 0.5 * ((float(price) - m20) / m20) + 0.3 * ((float(price) - m50) / m50) + 0.2 * ((float(price) - m200) / m200)
                    mf = round(mf, 4)
                except Exception:
                    mf = ''
            try:
                mapped_score = float(mapped_score_val) if mapped_score_val not in (None, '', 'N/A', 'yahoo') else ''
                if vpt is not None and mapped_score != '' and mf != '':
                    s_6 = (10 ** -4) * (vpt * price) * (1 + 2 * mf)
                    s_6 = round(s_6, 3)
                else:
                    s_6 = ''
            except Exception:
                s_6 = ''
        # Calculate s_8
        s_8_val = None
        try:
            s_6_val = float(s_6) if s_6 not in (None, '', 'N/A') else 0.0
            s_8_val = 2 * norm.cdf(s_6_val) - 1
        except Exception:
            s_8_val = None
        s_8_list.append(s_8_val)
        print_table_data.append([
            row[0] if len(row) > 0 else '',
            row[1] if len(row) > 1 else '',
            symbol,
            row[4] if len(row) > 4 else '',
            title,
            mapped_score_val,
            s_8_val
        ])
        if i == 0:
            try:
                phi = 2 * norm.cdf(s_6_val) - 1
                t = (2 / np.pi) * np.arctan(s_6_val)
                avg_phi_t = (phi + t) / 2
            except Exception:
                phi = None
                t = None
                avg_phi_t = None
    print(f"\nProcessed {len(table_data)} articles.")
    print("Results:")
    if print_table_data and any(any(str(cell).strip() for cell in row) for row in print_table_data):
        print(tabulate(print_table_data, headers=print_headers, tablefmt="grid"))
    else:
        print("No valid data to display in the main table.")
    print("\n The data table has been updated.")
    if table_data:
        df = pd.DataFrame(print_table_data, columns=headers)
        prev_lines = []
        if os.path.exists('sentiment_results.csv'):
            try:
                with open('sentiment_results.csv', 'r', encoding='utf-8') as f:
                    prev_lines = f.readlines()
            except Exception:
                prev_lines = []
        separator = '_' * 20 + '\n'
        from io import StringIO
        output = StringIO()
        df.to_csv(output, index=False, header=not prev_lines)
        new_data = output.getvalue()
        with open('sentiment_results.csv', 'w', encoding='utf-8', newline='') as f:
            if prev_lines:
                for line in prev_lines:
                    f.write(line)
                f.write(separator)
                f.write(new_data if not new_data.startswith('#') else new_data.split('\n',1)[1])
            else:
                f.write(new_data)
        show_raw = input("Would you like to see the raw data table? (y/n): ").strip().lower() == 'y'
        if show_raw:
            def format_large(val):
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
            raw_headers = [
                "Symbol", "Stock Price", "Volume", "Number of Trades", "Impact", "AD20", "AD50", "AD200", "M_f"
            ]
            raw_table_data = []
            for row in table_data:
                symbol = row[2]
                price = row[7] if len(row) > 7 and row[7] not in (None, '', 'N/A') else None
                volume = row[8] if len(row) > 8 else ''
                trades = row[9] if len(row) > 9 else ''
                vpt_str = row[3]
                ad20 = ad50 = ad200 = mf = ''
                vpt = None
                try:
                    vpt = float(vpt_str.replace(',', '')) if isinstance(vpt_str, str) and vpt_str else float(vpt_str)
                except Exception:
                    vpt = None
                if symbol and price not in (None, '', 'N/A'):
                    closes = closes_cache.get(symbol, [])
                    ma = ma_cache.get(symbol, {20: '', 50: '', 200: ''})
                    ma20, ma50, ma200 = ma[20], ma[50], ma[200]
                    try:
                        ad20 = round(float(price) - ma20, 3) if ma20 not in (None, 0, '') else ''
                    except Exception:
                        ad20 = ''
                    try:
                        ad50 = round(float(price) - ma50, 3) if ma50 not in (None, 0, '') else ''
                    except Exception:
                        ad50 = ''
                    try:
                        ad200 = round(float(price) - ma200, 3) if ma200 not in (None, 0, '') else ''
                    except Exception:
                        ad200 = ''
                    if all(isinstance(x, (int, float)) and x not in (None, 0) for x in [ma20, ma50, ma200]):
                        try:
                            mf = 0.5 * ((float(price) - ma20) / ma20) + 0.3 * ((float(price) - ma50) / ma50) + 0.2 * ((float(price) - ma200) / ma200)
                            mf = round(mf, 4)
                        except Exception:
                            mf = ''
                    else:
                        mf = ''
                raw_table_data.append([
                    symbol,
                    price,
                    format_large(volume),
                    format_large(trades),
                    vpt_str,
                    ad20,
                    ad50,
                    ad200,
                    mf
                ])
            if raw_table_data and any(any(str(cell).strip() for cell in row) for row in raw_table_data):
                print("\nRaw Data Table:")
                print(tabulate(raw_table_data, headers=raw_headers, tablefmt="grid"))
            else:
                print("No valid data to display in the raw data table.")
    else:
        print("No data to write to CSV.")


if __name__ == "__main__":
    main()