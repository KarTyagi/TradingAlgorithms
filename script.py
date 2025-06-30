import os
import re
import sys
import pandas as pd
from tabulate import tabulate
from datetime import datetime
from sentiment import initialize_analyzer, initialize_absa, analyze_sentiment, analyze_aspect_sentiment, sentiment_score
from news import scrape_yf, fetch_article_text, is_relevant_article
from stocks import get_stock_price_and_metrics, extract_sp500_mentions, format_volume

def main():
    analyzer = initialize_analyzer()
    absa_pipeline = initialize_absa()
    target_date = datetime.now().date()
    print(f"Fetching articles from Yahoo Finance for {target_date}...")
    articles = scrape_yf(target_date)
    print(f"[DEBUG] scrape_yf returned {len(articles)} articles.")
    if not articles:
        print("No articles available to analyze. Check site structure or date availability.")
        return
    include_non_stock = input("Include articles that do not mention any stocks? (y/n): ").strip().lower() == 'y'
    attempt = 1
    if os.path.exists('sentiment_results.csv'):
        try:
            prev_df = pd.read_csv('sentiment_results.csv')
            if 'Attempt' in prev_df.columns:
                attempt = prev_df['Attempt'].max() + 1
        except Exception:
            pass
    table_data = []
    for idx, article in enumerate(articles, 1):
        title = article["title"]
        print(f"[DEBUG] Article {idx}: {title}")
        mentions = extract_sp500_mentions(title)
        print(f"[DEBUG] Mentions found: {mentions}")
        article_text = fetch_article_text(article["link"])
        short_date = str(target_date)[2:]
        if not mentions:
            if not include_non_stock:
                continue
            print("[DEBUG] No symbol/company found, analyzing sentiment of article only.")
            sentiment_result = analyze_sentiment(analyzer, article_text or title)
            if 'error' in sentiment_result:
                confidence = ""
                score = ""
                sentiment_sign = "error"
            else:
                confidence = f"{sentiment_result['score']:.2%}"
                score = sentiment_score(sentiment_result["label"], sentiment_result["score"])
                label = sentiment_result["label"].lower()
                if label == "positive":
                    sentiment_sign = "+"
                elif label == "negative":
                    sentiment_sign = "-"
                elif label == "neutral":
                    sentiment_sign = "0"
                else:
                    sentiment_sign = label

            table_data.append([
                idx,
                short_date,
                "",
                "",
                "",      # Impact (v/t)
                "",      # Company Name
                "yahoo",
                (title[:30] + "...") if len(title) > 33 else title,
                sentiment_sign,
                confidence,
                score
            ])
            
            continue
        for symbol, company in mentions:
            print(f"[DEBUG] Matched symbol: {symbol}, company: {company}")
            stock_price, stock_volume, stock_trades = get_stock_price_and_metrics(symbol)
            # Calculate volume per trade (v/t)
            try:
                vpt = float(stock_volume) / float(stock_trades) if float(stock_trades) > 0 else 0
            except Exception:
                vpt = 0
            vpt_str = f"{vpt:,.1f}" if vpt else ""
            # Use more context for ABSA: prefer article_text, fallback to title
            context = article_text if article_text else title
            # Keyword-based sentiment adjustment
            keywords_positive = ["upgrade", "buy", "raise", "outperform", "beat", "growth", "profit"]
            keywords_negative = ["layoff", "cut", "downgrade", "sell", "underperform", "miss", "loss", "decline"]
            context_lower = context.lower()
            keyword_sentiment = None
            if any(word in context_lower for word in keywords_positive):
                keyword_sentiment = "positive"
            elif any(word in context_lower for word in keywords_negative):
                keyword_sentiment = "negative"
            try:
                label, raw_score, mapped_score = analyze_aspect_sentiment(absa_pipeline, context, symbol)
                # If ABSA is neutral, try fallback to keyword or regular sentiment
                if label == "neutral":
                    if keyword_sentiment:
                        label = keyword_sentiment
                        mapped_score = 1.0 if keyword_sentiment == "positive" else -1.0
                    else:
                        # Fallback to regular sentiment analyzer
                        reg_result = analyze_sentiment(analyzer, context)
                        reg_label = reg_result.get("label", "neutral").lower()
                        if reg_label in ("positive", "negative"):
                            label = reg_label
                            mapped_score = 1.0 if reg_label == "positive" else -1.0
                print(f"[DEBUG] Sentiment for {symbol}: label={label}, raw_score={raw_score}, mapped_score={mapped_score}")
                confidence = f"{raw_score:.2%}" if isinstance(raw_score, (int, float)) else ""
                # Fix: mapped_score should be a float, not a percent string
                mapped_score_val = round(mapped_score, 3) if isinstance(mapped_score, (int, float)) else ""
                score_val = str(round((mapped_score * vpt) / 100, 3)) if isinstance(mapped_score, (int, float)) else ""
                if label == "positive":
                    sentiment_sign = "+"
                elif label == "negative":
                    sentiment_sign = "-"
                elif label == "neutral":
                    sentiment_sign = "0"
                else:
                    sentiment_sign = label
            except Exception as e:
                print(f"[DEBUG] Sentiment analysis error for {symbol}: {e}")
                sentiment_sign = "error"
                confidence = "N/A"
                score = "N/A"
            table_data.append([
                idx,                # 0
                short_date,         # 1
                symbol,             # 2
                vpt_str,            # 3 (Impact)
                company,            # 4
                (title[:30] + "...") if len(title) > 33 else title,  # 5 (Article Title)
                mapped_score_val,   # 6 (Mapped Score)
                score_val,          # 7 (Score)
                stock_price,        # 8 (Stock Price)
                stock_volume,       # 9 (Volume)
                stock_trades        # 10 (Number of Trades)
            ])
    headers = ["#", "Date", "Symbol", "Impact", "Company Name", "Article Title", "Mapped Score", "Score"]
    print_headers = headers
    print_table_data = [
        [row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]]
        for row in table_data
    ]
    print(f"\nProcessed {len(table_data)} articles.")
    print("Results:")
    print(tabulate(print_table_data, headers=print_headers, tablefmt="grid"))
    if table_data:
        df = pd.DataFrame(print_table_data, columns=headers)
        if os.path.exists('sentiment_results.csv'):
            try:
                prev_lines = []
                with open('sentiment_results.csv', 'r', encoding='utf-8') as f:
                    prev_lines = f.readlines()
            except Exception:
                prev_lines = []
        else:
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
        print("\n The data table has been updated.")
        # Ask user if they want to see the raw data table
        show_raw = input("Would you like to see the raw data table including stock price, volume, and number of trades? (y/n): ").strip().lower() == 'y'
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
            raw_headers = ["Symbol", "Stock Price", "Volume", "Number of Trades"]
            raw_table_data = [
                [
                    row[2],
                    row[8],
                    format_large(row[9]),
                    format_large(row[10])
                ]
                for row in table_data
            ]
            print("\nRaw Data Table:")
            print(tabulate(raw_table_data, headers=raw_headers, tablefmt="grid"))
    else:
        print("No data to write to CSV.")

if __name__ == "__main__":
    main()