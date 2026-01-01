"""
Tools module for EMIXbot - Converts script.py functionality into separate tools
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import norm
import requests
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

from langchain_core.tools import tool
from sentiment import initialize_analyzer, initialize_absa, analyze_sentiment, analyze_aspect_sentiment
from news import scrape_yf, fetch_article_text
from stocks import get_stock_price_and_metrics, extract_sp500_mentions
from tabulate import tabulate


# ==================== Historical Data Tools ====================

# Internal implementation
def _get_fmp_historical_closes_impl(symbol: str, days: int = 200) -> list:
    """Internal implementation for fetching historical closes"""
    api_key = "SlOrGMWxbNwTiJGDfIrkswILZdxmv6O3"
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
        return []

@tool
def get_fmp_historical_closes(symbol: str, days: int = 200) -> list:
    """
    Fetch historical closing prices for a stock from Financial Modeling Prep API.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')
        days: Number of days of historical data to fetch (default: 200)
    
    Returns:
        List of closing prices in chronological order
    """
    return _get_fmp_historical_closes_impl(symbol, days)


@tool
def calculate_moving_averages(symbol: str, days: int = 200) -> dict:
    """
    Calculate moving averages (20, 50, 200 days) for a stock.
    
    Args:
        symbol: Stock ticker symbol
        days: Number of historical days to consider
    
    Returns:
        Dictionary with moving averages {20: value, 50: value, 200: value}
    """
    closes = _get_fmp_historical_closes_impl(symbol, days)
    ma = {}
    windows = [20, 50, 200]
    
    for w in windows:
        if len(closes) >= w:
            ma[w] = round(sum(closes[-w:]) / w, 2)
        else:
            ma[w] = None
    
    return ma


# ==================== Sentiment Analysis Tools ====================

# Cache sentiment models to avoid reinitializing
_sentiment_analyzer = None
_absa_pipeline = None

def _get_sentiment_models():
    """Get or initialize sentiment models (cached)"""
    global _sentiment_analyzer, _absa_pipeline
    
    if _sentiment_analyzer is None:
        try:
            _sentiment_analyzer = initialize_analyzer()
        except Exception as e:
            print(f"Warning: Could not initialize sentiment analyzer: {e}")
            _sentiment_analyzer = None
    
    if _absa_pipeline is None:
        try:
            _absa_pipeline = initialize_absa()
        except Exception as e:
            print(f"Warning: Could not initialize ABSA pipeline: {e}")
            _absa_pipeline = None
    
    return _sentiment_analyzer, _absa_pipeline


# Internal implementation
def _analyze_article_sentiment_impl(article_text: str, symbol: str) -> dict:
    """Internal implementation for analyzing article sentiment"""
    try:
        analyzer, absa_pipeline = _get_sentiment_models()
        
        keywords_positive = ["upgrade", "buy", "raise", "outperform", "beat", "growth", "profit", "merge", "acquisition"]
        keywords_negative = ["layoff", "cut", "downgrade", "sell", "underperform", "miss", "loss", "decline", "lawsuit", "fail"]
        
        context_lower = article_text.lower() if article_text else ""
        keyword_sentiment = None
        
        if any(word in context_lower for word in keywords_positive):
            keyword_sentiment = "positive"
        elif any(word in context_lower for word in keywords_negative):
            keyword_sentiment = "negative"
        
        # Try ABSA pipeline if available
        label = "neutral"
        raw_score = 0
        mapped_score = 0
        
        if absa_pipeline:
            try:
                label, raw_score, mapped_score = analyze_aspect_sentiment(absa_pipeline, article_text, symbol)
            except Exception as e:
                print(f"ABSA analysis failed for {symbol}: {e}")
                label = "neutral"
        
        # Fallback logic
        if label == "neutral":
            if keyword_sentiment:
                label = keyword_sentiment
                mapped_score = 1.0 if keyword_sentiment == "positive" else -1.0
            elif analyzer:
                try:
                    reg_result = analyze_sentiment(analyzer, article_text)
                    reg_label = reg_result.get("label", "neutral").lower()
                    if reg_label in ("positive", "negative"):
                        label = reg_label
                        mapped_score = 1.0 if reg_label == "positive" else -1.0
                except Exception as e:
                    print(f"Regular sentiment analysis failed for {symbol}: {e}")
        
        return {
            "symbol": symbol,
            "label": label,
            "raw_score": raw_score,
            "mapped_score": round(mapped_score, 3) if isinstance(mapped_score, (int, float)) else 0
        }
    except Exception as e:
        print(f"Error in sentiment analysis for {symbol}: {e}")
        return {
            "symbol": symbol,
            "label": "neutral",
            "raw_score": 0,
            "mapped_score": 0,
            "error": str(e)
        }

@tool
def analyze_article_sentiment(article_text: str, symbol: str) -> dict:
    """
    Analyze sentiment of an article for a specific stock symbol.
    
    Args:
        article_text: The article content to analyze
        symbol: Stock ticker symbol
    
    Returns:
        Dictionary with sentiment analysis results
    """
    return _analyze_article_sentiment_impl(article_text, symbol)


# ==================== Technical Analysis Tools ====================

# Internal implementation
def _calculate_momentum_factor_impl(symbol: str, price: float) -> dict:
    """Internal implementation for calculating momentum factor"""
    closes = _get_fmp_historical_closes_impl(symbol)
    ma = {}
    windows = [20, 50, 200]
    
    for w in windows:
        if len(closes) >= w:
            ma[w] = round(sum(closes[-w:]) / w, 2)
        else:
            ma[w] = None
    
    m20, m50, m200 = ma[20], ma[50], ma[200]
    
    mf = None
    if all(isinstance(x, (int, float)) and x not in (None, 0) for x in [m20, m50, m200]):
        try:
            mf = 0.5 * ((float(price) - m20) / m20) + 0.3 * ((float(price) - m50) / m50) + 0.2 * ((float(price) - m200) / m200)
            mf = round(mf, 4)
        except Exception:
            mf = None
    
    return {
        "symbol": symbol,
        "price": price,
        "ma20": m20,
        "ma50": m50,
        "ma200": m200,
        "momentum_factor": mf
    }

@tool
def calculate_momentum_factor(symbol: str, price: float) -> dict:
    """
    Calculate momentum factor (M_f) based on moving averages.
    
    Args:
        symbol: Stock ticker symbol
        price: Current stock price
    
    Returns:
        Dictionary with momentum factor and moving averages
    """
    return _calculate_momentum_factor_impl(symbol, price)


@tool
def calculate_s6_score(vpt: float, price: float, mf: float) -> dict:
    """
    Calculate S6 score: (10^-4) * (vpt * price) * (1 + 2*mf)
    
    Args:
        vpt: Volume Per Trade
        price: Stock price
        mf: Momentum factor
    
    Returns:
        Dictionary with S6 score and related metrics
    """
    try:
        s_6 = (10 ** -4) * (vpt * price) * (1 + 2 * mf)
        s_6 = round(s_6, 3)
        return {"s6": s_6, "success": True}
    except Exception as e:
        return {"s6": None, "success": False, "error": str(e)}


@tool
def calculate_s8_score(s_6: float) -> dict:
    """
    Calculate S8 score using normal CDF: 2 * norm.cdf(s_6) - 1
    
    Args:
        s_6: S6 score value
    
    Returns:
        Dictionary with S8 score
    """
    try:
        s_8 = 2 * norm.cdf(s_6) - 1
        return {
            "s8": round(s_8, 4),
            "s6": s_6,
            "success": True
        }
    except Exception as e:
        return {"s8": None, "success": False, "error": str(e)}


# ==================== News Scraping Tools ====================

# Internal implementation
def _scrape_latest_news_impl(target_date: str = None) -> list:
    """Internal implementation for scraping news"""
    if target_date is None:
        target_date = datetime.now().date()
    else:
        try:
            target_date = datetime.strptime(target_date, '%Y-%m-%d').date()
        except Exception:
            target_date = datetime.now().date()
    
    try:
        articles = scrape_yf(target_date)
        return articles[:25]  # Limit to 25 articles
    except Exception as e:
        return []

@tool
def scrape_latest_news(target_date: str = None) -> list:
    """
    Scrape latest financial news articles from Yahoo Finance.
    
    Args:
        target_date: Date to scrape news for (format: YYYY-MM-DD, default: today)
    
    Returns:
        List of articles with title and link
    """
    return _scrape_latest_news_impl(target_date)


# Internal implementation
def _extract_stock_mentions_impl(text: str) -> list:
    """Internal implementation for extracting stock mentions"""
    try:
        mentions = extract_sp500_mentions(text)
        return mentions
    except Exception as e:
        return []

@tool
def extract_stock_mentions(text: str) -> list:
    """
    Extract S&P 500 stock mentions from text.
    
    Args:
        text: Text to analyze (article title or content)
    
    Returns:
        List of tuples [(symbol, company_name), ...]
    """
    return _extract_stock_mentions_impl(text)


# ==================== Stock Data Tools ====================

# Internal implementation
def _get_stock_metrics_impl(symbol: str) -> dict:
    """Internal implementation for getting stock metrics"""
    try:
        stock_price, stock_volume, stock_trades = get_stock_price_and_metrics(symbol)
        vpt = float(stock_volume) / float(stock_trades) if float(stock_trades) > 0 else 0
        
        return {
            "symbol": symbol,
            "price": stock_price,
            "volume": stock_volume,
            "trades": stock_trades,
            "vpt": round(vpt, 1)
        }
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}

@tool
def get_stock_metrics(symbol: str) -> dict:
    """
    Get current stock price and trading metrics.
    
    Args:
        symbol: Stock ticker symbol
    
    Returns:
        Dictionary with price, volume, and trade count
    """
    try:
        stock_price, stock_volume, stock_trades = get_stock_price_and_metrics(symbol)
        vpt = float(stock_volume) / float(stock_trades) if float(stock_trades) > 0 else 0
        
        return {
            "symbol": symbol,
            "price": stock_price,
            "volume": stock_volume,
            "trades": stock_trades,
            "vpt": round(vpt, 1)
        }
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}


# ==================== Integrated Analysis Tools ====================

# Internal implementation
def _analyze_stock_impact_impl(symbol: str, price: float, volume: float, trades: float, mapped_score: float) -> dict:
    """Internal implementation for analyzing stock impact"""
    try:
        # Calculate VPT
        vpt = float(volume) / float(trades) if float(trades) > 0 else 0
        
        # Get momentum factor
        momentum_data = _calculate_momentum_factor_impl(symbol, price)
        mf = momentum_data.get("momentum_factor", 0)
        
        # Calculate S6
        if vpt and mf is not None:
            s_6 = (10 ** -4) * (vpt * price) * (1 + 2 * mf)
            s6 = round(s_6, 3)
            
            # Calculate S8
            s_8 = 2 * norm.cdf(s_6) - 1
            s8 = round(s_8, 4)
        else:
            s6 = s8 = 0
        
        return {
            "symbol": symbol,
            "price": price,
            "vpt": round(vpt, 1),
            "sentiment_score": mapped_score,
            "momentum_factor": mf,
            "s6_score": s6,
            "s8_score": s8,
            "ma20": momentum_data.get("ma20"),
            "ma50": momentum_data.get("ma50"),
            "ma200": momentum_data.get("ma200")
        }
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}

@tool
def analyze_stock_impact(symbol: str, price: float, volume: float, trades: float, mapped_score: float) -> dict:
    """
    Comprehensive analysis of stock impact combining all metrics.
    
    Args:
        symbol: Stock ticker symbol
        price: Current stock price
        volume: Trading volume
        trades: Number of trades
        mapped_score: Sentiment mapped score (-1 to 1)
    
    Returns:
        Dictionary with complete impact analysis
    """
    return _analyze_stock_impact_impl(symbol, price, volume, trades, mapped_score)


@tool
def generate_daily_analysis_report(target_date: str = None) -> dict:
    """
    Generate comprehensive daily analysis report for all stocks mentioned in news.
    
    Args:
        target_date: Date for analysis (format: YYYY-MM-DD, default: today)
    
    Returns:
        Dictionary with analysis results and statistics
    """
    if target_date is None:
        target_date = datetime.now().date()
    else:
        target_date = datetime.strptime(target_date, '%Y-%m-%d').date()
    
    try:
        articles = _scrape_latest_news_impl(str(target_date))
        
        if not articles:
            return {"status": "no_articles", "date": str(target_date)}
        
        results = []
        
        for idx, article in enumerate(articles, 1):
            title = article.get("title", "")
            link = article.get("link", "")
            
            mentions = _extract_stock_mentions_impl(title)
            if not mentions:
                continue
            
            article_text = fetch_article_text(link) if link else title
            
            for symbol, company in mentions:
                stock_metrics = _get_stock_metrics_impl(symbol)
                if "error" in stock_metrics:
                    continue
                
                sentiment_data = _analyze_article_sentiment_impl(article_text, symbol)
                mapped_score = sentiment_data.get("mapped_score", 0)
                
                impact = _analyze_stock_impact_impl(
                    symbol,
                    float(stock_metrics.get("price", 0)),
                    float(stock_metrics.get("volume", 0)),
                    float(stock_metrics.get("trades", 1)),
                    float(mapped_score) if isinstance(mapped_score, (int, float)) else 0
                )
                
                results.append({
                    "article_index": idx,
                    "title": title[:50] + "..." if len(title) > 50 else title,
                    **impact,
                    **sentiment_data
                })
        
        return {
            "date": str(target_date),
            "total_articles": len(articles),
            "analyzed_symbols": len(results),
            "results": results
        }
    
    except Exception as e:
        return {"status": "error", "error": str(e)}


# Internal implementation
def _run_complete_trading_analysis_impl(target_date: str = None) -> dict:
    """Internal implementation for complete trading analysis"""
    if target_date is None:
        target_date = datetime.now().date()
    else:
        try:
            target_date = datetime.strptime(target_date, '%Y-%m-%d').date()
        except Exception:
            target_date = datetime.now().date()
    
    try:
        # Step 1: Scrape latest news
        articles = _scrape_latest_news_impl(target_date=str(target_date))
        
        if not articles:
            return {
                "status": "no_articles",
                "date": str(target_date),
                "message": "No articles available to analyze. Check site structure or date availability."
            }
        
        print(f"[DEBUG] Scrape returned {len(articles)} articles")
        
        # Step 2: Process each article and extract stock mentions
        table_data = []
        closes_cache = {}
        ma_cache = {}
        
        for idx, article in enumerate(articles, 1):
            title = article.get("title", "")
            link = article.get("link", "")
            
            print(f"[DEBUG] Article {idx}: {title}")
            
            # Extract S&P 500 mentions from title
            mentions = _extract_stock_mentions_impl(title)
            print(f"[DEBUG] Mentions found: {mentions}")
            
            if not mentions:
                continue
            
            # Fetch full article text
            article_text = fetch_article_text(link) if link else title
            short_date = str(target_date)[2:]
            
            # Step 3: For each mentioned stock, analyze sentiment and metrics
            for symbol, company in mentions:
                print(f"[DEBUG] Processing symbol: {symbol}, company: {company}")
                
                # Get current stock metrics
                stock_metrics = _get_stock_metrics_impl(symbol)
                if "error" in stock_metrics:
                    continue
                
                stock_price = stock_metrics.get("price")
                stock_volume = stock_metrics.get("volume")
                stock_trades = stock_metrics.get("trades")
                vpt = stock_metrics.get("vpt", 0)
                vpt_str = f"{vpt:,.1f}" if vpt else ""
                
                # Step 4: Analyze sentiment
                sentiment_data = _analyze_article_sentiment_impl(article_text, symbol)
                mapped_score_val = sentiment_data.get("mapped_score", "")
                print(f"[DEBUG] Sentiment for {symbol}: {sentiment_data}")
                
                # Add to table data
                table_data.append({
                    "article_idx": idx,
                    "date": short_date,
                    "symbol": symbol,
                    "vpt": vpt_str,
                    "company": company,
                    "title": (title[:30] + "...") if len(title) > 33 else title,
                    "mapped_score": mapped_score_val,
                    "stock_price": stock_price,
                    "stock_volume": stock_volume,
                    "stock_trades": stock_trades,
                    "article_text": article_text
                })
        
        if not table_data:
            return {
                "status": "no_stocks_found",
                "date": str(target_date),
                "total_articles": len(articles),
                "message": "No stocks found in articles."
            }
        
        print(f"\n[DEBUG] Processing {len(table_data)} stock mentions")
        
        # Step 5: Calculate technical indicators and impact scores
        processed_results = []
        s_8_list = []
        
        for row in table_data:
            symbol = row["symbol"]
            price = row["stock_price"]
            vpt_str = row["vpt"]
            mapped_score_val = row["mapped_score"]
            
            # Initialize values
            s_6 = ""
            s_8_val = None
            
            # Parse VPT
            vpt = None
            try:
                vpt = float(vpt_str.replace(',', '')) if isinstance(vpt_str, str) and vpt_str else float(vpt_str)
            except Exception:
                vpt = None
            
            # Validate row
            valid_row = bool(symbol and price not in (None, '', 'N/A'))
            
            if valid_row:
                # Get or fetch historical closes
                if symbol not in closes_cache:
                    closes_cache[symbol] = _get_fmp_historical_closes_impl(symbol, days=200)
                
                # Calculate moving averages
                if symbol not in ma_cache:
                    closes = closes_cache[symbol]
                    ma = {}
                    windows = [20, 50, 200]
                    for w in windows:
                        if len(closes) >= w:
                            ma[w] = round(sum(closes[-w:]) / w, 2)
                        else:
                            ma[w] = None
                    ma_cache[symbol] = ma
                
                ma = ma_cache[symbol]
                m20, m50, m200 = ma[20], ma[50], ma[200]
                
                # Calculate momentum factor
                mf = ""
                if all(isinstance(x, (int, float)) and x not in (None, 0) for x in [m20, m50, m200]):
                    try:
                        mf = 0.5 * ((float(price) - m20) / m20) + 0.3 * ((float(price) - m50) / m50) + 0.2 * ((float(price) - m200) / m200)
                        mf = round(mf, 4)
                    except Exception:
                        mf = ""
                
                # Calculate S6 score
                try:
                    mapped_score = float(mapped_score_val) if mapped_score_val not in (None, '', 'N/A') else 0
                    if vpt is not None and mapped_score != '' and mf != '':
                        s_6 = (10 ** -4) * (vpt * price) * (1 + 2 * mf)
                        s_6 = round(s_6, 3)
                except Exception:
                    s_6 = ""
            
            # Calculate S8 score
            try:
                s_6_val = float(s_6) if s_6 not in (None, '', 'N/A') else 0.0
                s_8_val = 2 * norm.cdf(s_6_val) - 1
                s_8_val = round(s_8_val, 4) if s_8_val else None
            except Exception:
                s_8_val = None
            
            s_8_list.append(s_8_val)
            
            # Add processed row
            processed_results.append({
                "article_index": row["article_idx"],
                "date": row["date"],
                "symbol": symbol,
                "company": row["company"],
                "title": row["title"],
                "sentiment_score": mapped_score_val,
                "stock_price": price,
                "stock_volume": row["stock_volume"],
                "stock_trades": row["stock_trades"],
                "vpt": vpt,
                "ma20": ma_cache[symbol].get(20) if symbol in ma_cache else None,
                "ma50": ma_cache[symbol].get(50) if symbol in ma_cache else None,
                "ma200": ma_cache[symbol].get(200) if symbol in ma_cache else None,
                "momentum_factor": mf if isinstance(mf, float) else None,
                "s6_score": s_6 if isinstance(s_6, float) else None,
                "s8_score": s_8_val
            })
        
        # Step 6: Generate summary statistics
        positive_sentiments = sum(1 for r in processed_results if isinstance(r.get("sentiment_score"), (int, float)) and r["sentiment_score"] > 0)
        negative_sentiments = sum(1 for r in processed_results if isinstance(r.get("sentiment_score"), (int, float)) and r["sentiment_score"] < 0)
        neutral_sentiments = len(processed_results) - positive_sentiments - negative_sentiments
        
        # Get top performers by S8 score
        top_performers = sorted(
            [r for r in processed_results if r.get("s8_score") is not None],
            key=lambda x: x["s8_score"],
            reverse=True
        )[:5]
        
        # Get worst performers by S8 score
        worst_performers = sorted(
            [r for r in processed_results if r.get("s8_score") is not None],
            key=lambda x: x["s8_score"]
        )[:5]
        
        # Return comprehensive report
        return {
            "status": "success",
            "date": str(target_date),
            "summary": {
                "total_articles_analyzed": len(articles),
                "unique_stocks_found": len(processed_results),
                "sentiment_breakdown": {
                    "positive": positive_sentiments,
                    "negative": negative_sentiments,
                    "neutral": neutral_sentiments
                },
                "positive_percentage": round(100 * positive_sentiments / len(processed_results), 1) if processed_results else 0,
                "negative_percentage": round(100 * negative_sentiments / len(processed_results), 1) if processed_results else 0
            },
            "all_stocks": processed_results,
            "top_performers": top_performers,
            "worst_performers": worst_performers,
            "detailed_metrics": {
                "average_vpt": round(np.mean([r["vpt"] for r in processed_results if r.get("vpt")]), 2),
                "average_s8_score": round(np.mean([r["s8_score"] for r in processed_results if r.get("s8_score") is not None]), 4) if any(r.get("s8_score") is not None for r in processed_results) else None,
                "highest_s8": max([r["s8_score"] for r in processed_results if r.get("s8_score") is not None], default=None),
                "lowest_s8": min([r["s8_score"] for r in processed_results if r.get("s8_score") is not None], default=None)
            }
        }
    
    except Exception as e:
        return {
            "status": "error",
            "date": str(target_date),
            "error": str(e),
            "message": f"Error during trading analysis: {str(e)}"
        }

@tool
def run_complete_trading_analysis(target_date: str = None) -> dict:
    """
    Run complete trading analysis pipeline - Calls all underlying tools in a chain.
    Mimics the main() function from script.py
    
    This tool:
    1. Scrapes latest news articles
    2. Extracts stock mentions
    3. Analyzes sentiment for each stock
    4. Calculates technical indicators
    5. Computes impact scores (S6, S8)
    6. Generates comprehensive report
    
    Args:
        target_date: Date for analysis (format: YYYY-MM-DD, default: today)
    
    Returns:
        Dictionary with complete analysis results, statistics, and tables
    """
    return _run_complete_trading_analysis_impl(target_date)


# ==================== Utility Tools ====================

# Internal implementation
def _format_large_number_impl(value: float) -> str:
    """Internal implementation for formatting large numbers"""
    try:
        val = float(value)
        if val >= 1_000_000_000:
            return f"{val/1_000_000_000:.1f}B"
        elif val >= 1_000_000:
            return f"{val/1_000_000:.1f}M"
        elif val >= 1_000:
            return f"{val/1_000:.1f}K"
        else:
            return str(int(val))
    except Exception:
        return str(value)

@tool
def format_large_number(value: float) -> str:
    """
    Format large numbers with abbreviations (b, m, k).
    
    Args:
        value: Number to format
    
    Returns:
        Formatted string
    """
    return _format_large_number_impl(value)
