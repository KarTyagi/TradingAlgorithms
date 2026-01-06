import streamlit as st
import time
import os
import json
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from tools_module import (
    get_fmp_historical_closes,
    calculate_moving_averages,
    analyze_article_sentiment,
    calculate_momentum_factor,
    calculate_s6_score,
    calculate_s8_score,
    scrape_latest_news,
    extract_stock_mentions,
    get_stock_metrics,
    analyze_stock_impact,
    generate_daily_analysis_report,
    run_complete_trading_analysis,
    format_large_number,
    _scrape_latest_news_impl,
    _extract_stock_mentions_impl,
    _get_stock_metrics_impl,
    _format_large_number_impl,
    _run_complete_trading_analysis_impl
)

# Page configuration
st.set_page_config(
    page_title="EMIXbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ChatGPT-like styling
st.markdown("""
    <style>
    /* Main container */
    .main {
        background-color: #343541;
    }
    
    /* Chat container */
    .stChatFloatingInputContainer {
        background-color: #40414f;
    }
    
    /* User message styling */
    .stChatMessage[data-testid="user-message"] {
        background-color: #343541;
    }
    
    /* Assistant message styling */
    .stChatMessage[data-testid="assistant-message"] {
        background-color: #444654;
    }
    
    /* Input box styling */
    .stChatInputContainer > div {
        background-color: #40414f;
        border-radius: 8px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #202123;
    }
    
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #ffffff !important;
    }
    
    /* Force sidebar to always be visible */
    [data-testid="stSidebar"] {
        display: block !important;
        min-width: 300px;
        position: relative !important;
    }
    
    /* Button styling */
    .stButton button {
        background-color: #10a37f;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 8px 16px;
    }
    
    .stButton button:hover {
        background-color: #1a7f64;
    }
    
    /* Hamburger menu button - move to main page */
    [data-testid="stDecoration"] {
        visibility: visible !important;
        position: fixed !important;
        top: 10px;
        left: 10px;
        z-index: 1000 !important;
        background: none !important;
    }
    
    button[kind="header"] {
        visibility: visible !important;
    }
    
    /* Force sidebar to always be visible */
    [data-testid="stSidebar"] {
        background-color: #202123;
        display: block !important;
        min-width: 300px;
        position: relative !important;
    }
    
    /* Header styling */
    h1 {
        color: #ffffff;
        font-weight: 600;
    }
    
    /* Text color - all text white */
    p, div, span, label, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    /* Chat messages text */
    .stChatMessage p, .stChatMessage div, .stChatMessage span {
        color: #ffffff !important;
    }
    
    /* Input labels and text */
    .stTextInput label, .stSelectbox label, .stSlider label {
        color: #ffffff !important;
    }
    
    /* Markdown content */
    .stMarkdown, .stMarkdown p, .stMarkdown div {
        color: #ffffff !important;
    }
    
    /* Info, warning, error messages */
    .stAlert p, .stAlert div {
        color: #ffffff !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_tool" not in st.session_state:
    st.session_state.current_tool = None

# Sidebar with API configuration
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # API Configuration
    st.markdown("### Azure OpenAI Setup")
    api_key = st.text_input("API Key", type="password", key="api_key_input")
    deployment_name = st.text_input("Deployment Name", value="gpt-4", key="deployment_input")
    
    if api_key:
        os.environ["AZURE_OPENAI_API_KEY"] = api_key
    if deployment_name:
        os.environ["AZURE_DEPLOYMENT_NAME"] = deployment_name
    
    st.markdown("---")
    st.markdown("## 💬 Chat")
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_tool = None
        st.rerun()
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("EMIXbot - Your AI trading and quantitative analysis assistant.")
    st.markdown("---")
    st.markdown("### Available Tools")
    st.markdown("📈 Stock Price Lookup")
    st.markdown("⚖️ Portfolio Risk Calculator")
    st.markdown("💭 Market Sentiment Analysis")
    st.markdown("📊 Technical Analysis")
    st.markdown("---")
    st.markdown("### Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1, key="temp_slider")
    max_tokens = st.slider("Max Tokens", 100, 2000, 500, 100, key="tokens_slider")

# Check if API key is configured
if not os.getenv("AZURE_OPENAI_API_KEY", "").strip():
    st.error("❌ Azure OpenAI API Key Not Configured")
    st.info("Please enter your Azure OpenAI API key in the sidebar to continue.")
    st.stop()

# Define tools for the agent
@tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a given ticker symbol."""
    st.session_state.current_tool = "📈 Stock Price Lookup"
    try:
        metrics = get_stock_metrics(ticker)
        if "error" in metrics:
            return f"Error getting metrics for {ticker}: {metrics['error']}"
        return f"Stock: {ticker}\nPrice: ${metrics['price']}\nVolume: {format_large_number(metrics['volume'])}\nTrades: {format_large_number(metrics['trades'])}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def calculate_portfolio_risk(portfolio: str) -> str:
    """Calculate risk metrics for a given portfolio."""
    st.session_state.current_tool = "⚖️ Portfolio Risk Calculator"
    try:
        # Extract symbols from portfolio description
        symbols = extract_stock_mentions(portfolio)
        if not symbols:
            return "No valid stock symbols found in portfolio description."
        
        risk_data = []
        for symbol, _ in symbols:
            try:
                ma = calculate_moving_averages(symbol)
                metrics = get_stock_metrics(symbol)
                risk_data.append({
                    "symbol": symbol,
                    "price": metrics.get("price"),
                    "ma20": ma.get(20),
                    "ma50": ma.get(50),
                    "ma200": ma.get(200)
                })
            except:
                continue
        
        if not risk_data:
            return "Could not calculate risk metrics."
        
        result = "Portfolio Risk Analysis:\n\n"
        for data in risk_data:
            result += f"{data['symbol']}: Price ${data['price']}\n"
            result += f"  MA20: ${data['ma20']}, MA50: ${data['ma50']}, MA200: ${data['ma200']}\n"
        return result
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_market_sentiment(query: str) -> str:
    """Analyze market sentiment for a given topic or stock."""
    st.session_state.current_tool = "💭 Market Sentiment Analysis"
    try:
        # First try to get stock metrics directly if it's a ticker symbol
        ticker = query.upper().strip()
        if len(ticker) <= 5 and ticker.isalpha():
            try:
                metrics = get_stock_metrics(ticker)
                if "error" not in metrics:
                    # We have a valid ticker, analyze its sentiment from news
                    articles = scrape_latest_news()
                    if articles:
                        relevant_articles = [a for a in articles if ticker in a.get("title", "").upper()]
                        
                        if relevant_articles:
                            sentiment_scores = []
                            for article in relevant_articles[:5]:
                                title = article.get("title", "")
                                try:
                                    sentiment = analyze_article_sentiment(title, ticker)
                                    if sentiment.get("label") != "error":
                                        sentiment_scores.append(sentiment)
                                except Exception as e:
                                    print(f"Error analyzing article: {e}")
                                    continue
                            
                            if sentiment_scores:
                                positive = sum(1 for s in sentiment_scores if s.get("label") == "positive")
                                negative = sum(1 for s in sentiment_scores if s.get("label") == "negative")
                                neutral = len(sentiment_scores) - positive - negative
                                total = len(sentiment_scores)
                                
                                result = f"Market Sentiment for {ticker}:\n\n"
                                result += f"Analyzed {total} news articles\n"
                                result += f"Positive: {positive}/{total} ({100*positive/total:.0f}%)\n"
                                result += f"Negative: {negative}/{total} ({100*negative/total:.0f}%)\n"
                                result += f"Neutral: {neutral}/{total} ({100*neutral/total:.0f}%)\n\n"
                                
                                # Add overall assessment
                                if positive > negative:
                                    result += f"Overall: Bullish 📈"
                                elif negative > positive:
                                    result += f"Overall: Bearish 📉"
                                else:
                                    result += f"Overall: Mixed/Neutral ⚖️"
                                
                                return result
            except Exception as e:
                print(f"Direct ticker analysis failed: {e}")
        
        # Fallback to general query search
        articles = scrape_latest_news()
        if not articles:
            return f"No articles available for sentiment analysis. This could be because:\n- Markets are closed\n- No recent news available\n- Connection issues with news source"
        
        # Find relevant articles
        relevant_articles = [a for a in articles if query.lower() in a.get("title", "").lower()]
        
        if not relevant_articles:
            return f"No recent articles found mentioning '{query}'.\n\nTry:\n- Using the stock ticker (e.g., 'AAPL' instead of 'Apple')\n- A broader search term (e.g., 'tech stocks')\n- Checking if markets are open"
        
        sentiment_scores = []
        for article in relevant_articles[:10]:  # Analyze up to 10 articles
            title = article.get("title", "")
            mentions = extract_stock_mentions(title)
            
            # If no mentions, try analyzing sentiment for the query itself
            if not mentions:
                try:
                    sentiment = analyze_article_sentiment(title, query.upper())
                    if sentiment.get("label") != "error":
                        sentiment_scores.append(sentiment)
                except:
                    continue
            else:
                for symbol, _ in mentions:
                    try:
                        sentiment = analyze_article_sentiment(title, symbol)
                        if sentiment.get("label") != "error":
                            sentiment_scores.append(sentiment)
                    except:
                        continue
        
        if not sentiment_scores:
            return f"Could not analyze sentiment for '{query}'.\n\nThis may be due to:\n- Sentiment models not fully loaded\n- Article text unavailable\n- Query too vague\n\nTry a specific stock ticker instead."
        
        positive = sum(1 for s in sentiment_scores if s.get("label") == "positive")
        negative = sum(1 for s in sentiment_scores if s.get("label") == "negative")
        neutral = len(sentiment_scores) - positive - negative
        total = len(sentiment_scores)
        
        result = f"Market Sentiment for '{query}':\n\n"
        result += f"Analyzed {total} mentions across {len(relevant_articles)} articles\n"
        result += f"Positive: {positive}/{total} ({100*positive/total:.0f}%)\n"
        result += f"Negative: {negative}/{total} ({100*negative/total:.0f}%)\n"
        result += f"Neutral: {neutral}/{total} ({100*neutral/total:.0f}%)\n\n"
        
        # Add overall assessment
        if positive > negative:
            result += f"Overall: Bullish 📈"
        elif negative > positive:
            result += f"Overall: Bearish 📉"
        else:
            result += f"Overall: Mixed/Neutral ⚖️"
        
        return result
    except Exception as e:
        return f"Error analyzing market sentiment: {str(e)}\n\nPlease try:\n- A specific stock ticker (e.g., 'IBM', 'AAPL')\n- Checking your internet connection\n- Trying again in a few moments"
        return result
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def technical_analysis(ticker: str) -> str:
    """Perform technical analysis on a stock."""
    st.session_state.current_tool = "📊 Technical Analysis"
    try:
        metrics = get_stock_metrics(ticker)
        if "error" in metrics:
            return f"Error: {metrics['error']}"
        
        price = metrics['price']
        momentum = calculate_momentum_factor(ticker, price)
        
        result = f"Technical Analysis for {ticker}:\n\n"
        result += f"Current Price: ${price}\n"
        result += f"MA20: ${momentum['ma20']}\n"
        result += f"MA50: ${momentum['ma50']}\n"
        result += f"MA200: ${momentum['ma200']}\n"
        result += f"Momentum Factor: {momentum.get('momentum_factor', 'N/A')}\n"
        
        # Determine trend
        ma20 = momentum['ma20']
        ma50 = momentum['ma50']
        ma200 = momentum['ma200']
        
        if ma20 and ma50 and ma200:
            if price > ma20 > ma50 > ma200:
                result += "\nTrend: Strong Uptrend 📈"
            elif price < ma20 < ma50 < ma200:
                result += "\nTrend: Strong Downtrend 📉"
            else:
                result += "\nTrend: Mixed/Consolidation 📊"
        
        return result
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def generate_daily_report(date: str = None) -> str:
    """Generate comprehensive daily analysis report."""
    st.session_state.current_tool = "📊 Daily Report Generator"
    try:
        report = generate_daily_analysis_report(date)
        
        if report.get("status") == "no_articles":
            return f"No articles found for {report.get('date')}."
        
        if report.get("status") == "error":
            return f"Error: {report.get('error')}"
        
        results = report.get("results", [])
        result_text = f"Daily Analysis Report - {report.get('date')}\n\n"
        result_text += f"Total Articles Analyzed: {report.get('total_articles')}\n"
        result_text += f"Unique Symbols Found: {report.get('analyzed_symbols')}\n\n"
        
        for r in results[:10]:  # Show top 10
            result_text += f"\n{r.get('symbol')}: {r.get('title')}\n"
            result_text += f"  Price: ${r.get('price')}, S8 Score: {r.get('s8_score')}\n"
            result_text += f"  Sentiment: {r.get('label')}\n"
        
        return result_text
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def run_full_trading_analysis(date: str = None) -> str:
    """Run complete trading analysis pipeline - analyzes all news and stocks in one comprehensive chain with progress tracking."""
    st.session_state.current_tool = "🔗 Complete Trading Analysis"
    
    # Initialize progress tracking in session state
    if 'analysis_progress' not in st.session_state:
        st.session_state.analysis_progress = []
    st.session_state.analysis_progress = []
    
    try:
        progress_container = st.container()
        
        # Step 1: Scrape news articles
        with progress_container:
            with st.expander("🔄 STEP 1: Scraping News Articles", expanded=True):
                st.write("**Tool:** `scrape_latest_news()`")
                st.write(f"**Parameters:** date={date if date else 'today'}")
                with st.spinner("Fetching articles from Yahoo Finance..."):
                    articles = _scrape_latest_news_impl(date)
                if articles:
                    st.success(f"✅ **Output:** Found {len(articles)} articles")
                    st.write(f"Sample: {articles[0].get('title', 'N/A')[:80]}...")
                else:
                    st.error("❌ No articles found")
                    return "⚠️ No articles found. Cannot proceed with analysis."
        
        # Step 2: Extract stock mentions
        mentions_count = 0
        unique_stocks = set()
        with progress_container:
            with st.expander("🔄 STEP 2: Extracting Stock Mentions", expanded=True):
                st.write("**Tool:** `extract_stock_mentions()`")
                st.write(f"**Processing:** {len(articles)} articles")
                progress_bar = st.progress(0)
                for idx, article in enumerate(articles):
                    mentions = _extract_stock_mentions_impl(article.get('title', ''))
                    mentions_count += len(mentions)
                    for symbol, _ in mentions:
                        unique_stocks.add(symbol)
                    progress_bar.progress((idx + 1) / len(articles))
                st.success(f"✅ **Output:** Found {mentions_count} mentions across {len(unique_stocks)} unique stocks")
                if unique_stocks:
                    st.write(f"Stocks: {', '.join(sorted(list(unique_stocks)[:10]))}" + (" ..." if len(unique_stocks) > 10 else ""))
        
        if not unique_stocks:
            return "⚠️ No stock mentions found in articles."
        
        # Step 3: Fetch stock metrics
        stock_data = {}
        with progress_container:
            with st.expander("🔄 STEP 3: Fetching Stock Metrics", expanded=True):
                st.write("**Tool:** `get_stock_metrics()`")
                st.write(f"**Processing:** {len(unique_stocks)} stocks")
                progress_bar = st.progress(0)
                for idx, symbol in enumerate(sorted(unique_stocks)):
                    metrics = _get_stock_metrics_impl(symbol)
                    if "error" not in metrics:
                        stock_data[symbol] = metrics
                    progress_bar.progress((idx + 1) / len(unique_stocks))
                st.success(f"✅ **Output:** Retrieved metrics for {len(stock_data)} stocks")
                if stock_data:
                    sample_symbol = list(stock_data.keys())[0]
                    sample = stock_data[sample_symbol]
                    st.write(f"Sample ({sample_symbol}): Price=${sample.get('price')}, Volume={_format_large_number_impl(sample.get('volume', 0))}")
        
        # Step 4: Run complete analysis (backend)
        with progress_container:
            with st.expander("🔄 STEP 4: Running Complete Analysis Pipeline", expanded=True):
                st.write("**Executing:** Full trading analysis with sentiment, technical indicators, and scoring")
                with st.spinner("Processing all stocks..."):
                    result = _run_complete_trading_analysis_impl(date)
                
                # Check status
                status = result.get("status", "unknown")
                
                if status == "success":
                    summary = result.get("summary", {})
                    st.success(f"✅ **Output:** Analysis completed successfully")
                    st.write(f"• Analyzed {summary.get('unique_stocks_found', 0)} stocks")
                    st.write(f"• Positive: {summary.get('positive_percentage', 0):.1f}%, Negative: {summary.get('negative_percentage', 0):.1f}%")
                elif status == "no_articles":
                    st.error(f"❌ No articles found for {result.get('date')}")
                    return "⚠️ No articles available for analysis."
                elif status == "no_stocks_found":
                    st.warning(f"⚠️ Found articles but no stocks mentioned")
                    return f"Found {result.get('total_articles', 0)} articles but no S&P 500 stocks were mentioned."
                elif status == "error":
                    st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
                    return f"Error during analysis: {result.get('error')}"
                else:
                    st.warning(f"⚠️ Unexpected status: {status}")
                    return f"Unexpected status: {status}"
        
        # Step 5: Display results
        with progress_container:
            with st.expander("📊 STEP 5: Generating Final Report", expanded=True):
                if status != "success":
                    st.error("Cannot generate report - analysis did not complete successfully")
                else:
                    st.write("**Formatting comprehensive report...**")
                    
                    # Show detailed breakdown
                    all_stocks = result.get("all_stocks", [])
                    top_performers = result.get("top_performers", [])
                    worst_performers = result.get("worst_performers", [])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Stocks", len(all_stocks))
                    with col2:
                        sentiment = result.get("summary", {}).get("sentiment_breakdown", {})
                        st.metric("Positive", sentiment.get('positive', 0))
                    with col3:
                        st.metric("Negative", sentiment.get('negative', 0))
                    
                    st.success("✅ Report generation complete")
        
        # Return formatted report
        if status != "success":
            return f"Analysis incomplete: {result.get('message', result.get('error', 'Unknown error'))}"
        
        # Format the comprehensive report
        report_text = f"{'='*70}\n"
        report_text += f"📊 COMPLETE TRADING ANALYSIS REPORT - {result.get('date')}\n"
        report_text += f"{'='*70}\n\n"
        
        # Summary section
        summary = result.get("summary", {})
        report_text += f"📋 SUMMARY\n"
        report_text += f"• Total Articles Analyzed: {summary.get('total_articles_analyzed', 0)}\n"
        report_text += f"• Unique Stocks Found: {summary.get('unique_stocks_found', 0)}\n\n"
        
        # Sentiment breakdown
        sentiment = summary.get("sentiment_breakdown", {})
        report_text += f"💭 SENTIMENT BREAKDOWN\n"
        report_text += f"• Positive: {sentiment.get('positive', 0)} stocks ({summary.get('positive_percentage', 0):.1f}%)\n"
        report_text += f"• Negative: {sentiment.get('negative', 0)} stocks ({summary.get('negative_percentage', 0):.1f}%)\n"
        report_text += f"• Neutral: {sentiment.get('neutral', 0)} stocks\n\n"
        
        # Metrics
        metrics = result.get("detailed_metrics", {})
        report_text += f"📈 KEY METRICS\n"
        if metrics.get('average_vpt'):
            report_text += f"• Average VPT (Volume Per Trade): {metrics['average_vpt']:,.2f}\n"
        if metrics.get('average_s8_score') is not None:
            report_text += f"• Average S8 Score: {metrics['average_s8_score']:.4f}\n"
        if metrics.get('highest_s8') is not None:
            report_text += f"• Highest S8: {metrics['highest_s8']:.4f}\n"
        if metrics.get('lowest_s8') is not None:
            report_text += f"• Lowest S8: {metrics['lowest_s8']:.4f}\n"
        report_text += "\n"
        
        # Top performers
        top = result.get("top_performers", [])
        if top:
            report_text += f"🏆 TOP 5 PERFORMERS (Highest S8 Scores)\n"
            for i, stock in enumerate(top, 1):
                s8 = stock.get('s8_score', 'N/A')
                s8_str = f"{s8:.4f}" if isinstance(s8, (int, float)) else str(s8)
                price = stock.get('stock_price', 'N/A')
                price_str = f"${price:.2f}" if isinstance(price, (int, float)) else str(price)
                report_text += f"  {i}. {stock['symbol']:6} → S8: {s8_str:>8} | Price: {price_str:>8}\n"
            report_text += "\n"
        
        # Worst performers
        worst = result.get("worst_performers", [])
        if worst:
            report_text += f"📉 WORST 5 PERFORMERS (Lowest S8 Scores)\n"
            for i, stock in enumerate(worst, 1):
                s8 = stock.get('s8_score', 'N/A')
                s8_str = f"{s8:.4f}" if isinstance(s8, (int, float)) else str(s8)
                price = stock.get('stock_price', 'N/A')
                price_str = f"${price:.2f}" if isinstance(price, (int, float)) else str(price)
                report_text += f"  {i}. {stock['symbol']:6} → S8: {s8_str:>8} | Price: {price_str:>8}\n"
            report_text += "\n"
        
        # All stocks summary table
        all_stocks = result.get("all_stocks", [])
        if all_stocks:
            report_text += f"📋 DETAILED ANALYSIS ({len(all_stocks)} stocks)\n"
            report_text += f"{'-'*70}\n"
            report_text += f"{'SYMBOL':<8} {'PRICE':>10} {'SENTIMENT':>10} {'S8 SCORE':>12}\n"
            report_text += f"{'-'*70}\n"
            
            for stock in all_stocks[:15]:  # Show first 15
                symbol = stock.get('symbol', 'N/A')
                price = stock.get('stock_price', 'N/A')
                price_str = f"${price:.2f}" if isinstance(price, (int, float)) else str(price)
                sent = stock.get('sentiment_score', 'N/A')
                sent_str = f"{sent:.3f}" if isinstance(sent, (int, float)) else str(sent)
                s8 = stock.get('s8_score', 'N/A')
                s8_str = f"{s8:.4f}" if isinstance(s8, (int, float)) else str(s8)
                
                report_text += f"{symbol:<8} {price_str:>10} {sent_str:>10} {s8_str:>12}\n"
            
            if len(all_stocks) > 15:
                report_text += f"{'-'*70}\n"
                report_text += f"... and {len(all_stocks) - 15} more stocks analyzed\n"
        
        report_text += f"\n{'='*70}\n"
        report_text += "\n💡 TIP: S8 scores range from -1 to +1. Higher scores indicate stronger bullish signals.\n"
        
        return report_text
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        with progress_container:
            with st.expander("❌ ERROR DETAILS", expanded=True):
                st.error(f"**Error:** {str(e)}")
                st.code(error_details, language="python")
        return f"❌ Error running complete trading analysis: {str(e)}\n\nPlease try:\n- Individual tools instead (stock price, sentiment analysis)\n- Checking your API keys and network connection\n- Running during market hours for better data availability"

@tool
def test_pipeline_components() -> str:
    """Test individual components of the trading analysis pipeline to diagnose issues."""
    st.session_state.current_tool = "🔧 Pipeline Diagnostics"
    
    results = []
    
    # Test 1: News scraping
    results.append("=" * 50)
    results.append("TEST 1: News Scraping")
    results.append("-" * 50)
    try:
        articles = scrape_latest_news()
        if articles:
            results.append(f"✅ SUCCESS: Found {len(articles)} articles")
            results.append(f"   Sample title: {articles[0].get('title', 'N/A')[:60]}...")
        else:
            results.append("⚠️  WARNING: No articles found")
    except Exception as e:
        results.append(f"❌ FAILED: {str(e)}")
    
    # Test 2: Stock mention extraction
    results.append("\n" + "=" * 50)
    results.append("TEST 2: Stock Mention Extraction")
    results.append("-" * 50)
    try:
        test_text = "Apple Inc and Microsoft Corporation announced record earnings"
        mentions = extract_stock_mentions(test_text)
        if mentions:
            results.append(f"✅ SUCCESS: Found {len(mentions)} mentions")
            results.append(f"   Extracted: {', '.join([f'{s}({c})' for s, c in mentions])}")
        else:
            results.append("⚠️  WARNING: No mentions extracted from test text")
    except Exception as e:
        results.append(f"❌ FAILED: {str(e)}")
    
    # Test 3: Stock metrics API
    results.append("\n" + "=" * 50)
    results.append("TEST 3: Stock Metrics API (Polygon)")
    results.append("-" * 50)
    try:
        metrics = get_stock_metrics("AAPL")
        if "error" in metrics:
            results.append(f"⚠️  WARNING: API returned error: {metrics['error']}")
        elif metrics.get('price'):
            results.append(f"✅ SUCCESS: AAPL Price: ${metrics['price']}")
            results.append(f"   Volume: {format_large_number(metrics.get('volume', 0))}")
        else:
            results.append("⚠️  WARNING: No price data returned")
    except Exception as e:
        results.append(f"❌ FAILED: {str(e)}")
    
    # Test 4: Historical data API
    results.append("\n" + "=" * 50)
    results.append("TEST 4: Historical Data API (FMP)")
    results.append("-" * 50)
    try:
        closes = get_fmp_historical_closes("AAPL", days=50)
        if closes and len(closes) > 0:
            results.append(f"✅ SUCCESS: Retrieved {len(closes)} days of data")
            results.append(f"   Latest close: ${closes[-1]:.2f}")
        else:
            results.append("⚠️  WARNING: No historical data returned")
    except Exception as e:
        results.append(f"❌ FAILED: {str(e)}")
    
    # Test 5: Sentiment analysis
    results.append("\n" + "=" * 50)
    results.append("TEST 5: Sentiment Analysis")
    results.append("-" * 50)
    try:
        test_text = "Apple stock surged after strong earnings beat analyst expectations"
        sentiment = analyze_article_sentiment(test_text, "AAPL")
        results.append(f"✅ SUCCESS: Analysis completed")
        results.append(f"   Label: {sentiment.get('label', 'N/A')}")
        results.append(f"   Score: {sentiment.get('mapped_score', 'N/A')}")
    except Exception as e:
        results.append(f"❌ FAILED: {str(e)}")
    
    results.append("\n" + "=" * 50)
    results.append("DIAGNOSTIC SUMMARY")
    results.append("=" * 50)
    
    success_count = sum(1 for line in results if "✅ SUCCESS" in line)
    warning_count = sum(1 for line in results if "⚠️  WARNING" in line)
    fail_count = sum(1 for line in results if "❌ FAILED" in line)
    
    results.append(f"\nResults: {success_count} passed, {warning_count} warnings, {fail_count} failed")
    
    if fail_count > 0:
        results.append("\n⚠️  Some components failed. Check:")
        results.append("   • Network connectivity")
        results.append("   • API keys in .env file")
        results.append("   • Required packages installed")
    elif warning_count > 0:
        results.append("\n⚠️  Some warnings detected. Pipeline may work with reduced functionality.")
    else:
        results.append("\n✅ All components working! Pipeline should function normally.")
    
    return "\n".join(results)

# Create list of tools
tools = [
    get_stock_price,
    calculate_portfolio_risk,
    get_market_sentiment,
    technical_analysis,
    generate_daily_report,
    run_full_trading_analysis,
    test_pipeline_components
]

# Initialize Azure OpenAI with LangChain
@st.cache_resource
def get_llm():
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    
    # If no valid API key, return None (will use demo mode)
    if not api_key or api_key == "your-api-key-here":
        return None
    
    try:
        return AzureChatOpenAI(
            azure_endpoint="https://rnk-party.openai.azure.com/",
            api_key=api_key,
            api_version="2024-02-15-preview",
            deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4"),
            temperature=0.7,
        )
    except Exception as e:
        st.warning(f"Failed to initialize Azure OpenAI: {str(e)}")
        return None

# Custom agent executor
def execute_agent(llm, user_input, chat_history):
    """Execute agent with tool calling"""
    
    if llm is None:
        raise ValueError("Azure OpenAI API key is not configured. Please add a valid API key in the sidebar.")
    
    # Build system prompt with tools description
    tool_descriptions = "\n".join([
        f"- {tool_obj.name}: {tool_obj.description}" 
        for tool_obj in tools
    ])
    
    system_prompt = f"""You are EMIXbot, a helpful AI assistant specialized in trading, finance, and quantitative analysis. 

AVAILABLE TOOLS:
{tool_descriptions}

When a user query requires using a tool, respond with JSON in this format:
{{"tool_name": "tool_name_here", "tool_input": "input_here"}}

Otherwise, just respond with your helpful answer. Always explain which tool you're using and why."""

    # Build messages
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add chat history
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append({"role": "user", "content": msg["content"]})
        else:
            messages.append({"role": "assistant", "content": msg["content"]})
    
    # Add current user input
    messages.append({"role": "user", "content": user_input})
    
    try:
        response = llm.invoke(messages)
        response_text = response.content
        
        # Check if response contains tool call
        if "{" in response_text and "tool_name" in response_text:
            try:
                # Extract JSON from response
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_str = response_text[json_start:json_end]
                tool_call = json.loads(json_str)
                
                tool_name = tool_call.get("tool_name")
                tool_input = tool_call.get("tool_input")
                
                # Find and execute the tool
                selected_tool = None
                for t in tools:
                    if t.name == tool_name:
                        selected_tool = t
                        break
                
                if selected_tool:
                    st.session_state.current_tool = f"🔧 {tool_name.replace('_', ' ').title()}"
                    tool_result = selected_tool.invoke(tool_input)
                    
                    # Get follow-up response from LLM with tool result
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({"role": "user", "content": f"Tool result: {tool_result}\n\nNow provide your final response based on this tool result."})
                    
                    final_response = llm.invoke(messages)
                    return final_response.content
            except (json.JSONDecodeError, KeyError):
                return response_text
        else:
            return response_text
            
    except Exception as e:
        raise Exception(f"Error calling Azure OpenAI: {str(e)}")

# Main chat interface
st.title("EMIXbot")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Send a message..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response with tool usage
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        tool_placeholder = st.empty()
        full_response = ""
        
        try:
            # Reset current tool
            st.session_state.current_tool = None
            
            # Get LLM
            llm = get_llm()
            
            # Prepare chat history (exclude current message)
            chat_history = st.session_state.messages[:-1]
            
            # Show thinking indicator
            with st.spinner("🤔 Thinking..."):
                full_response = execute_agent(llm, prompt, chat_history)
            
            # Display which tool was used
            if st.session_state.current_tool:
                tool_placeholder.info(f"{st.session_state.current_tool}")
            
            # Display the response with streaming effect
            temp_response = ""
            for chunk in full_response.split():
                temp_response += chunk + " "
                time.sleep(0.02)
                message_placeholder.markdown(temp_response + "▌")
            
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            error_message = str(e)
            message_placeholder.error(error_message)
            full_response = error_message
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Display welcome message if no messages
if len(st.session_state.messages) == 0:
    st.markdown("""
        <div style='text-align: center; padding: 50px;'>
            <h2 style='color: #ffffff;'>Quant Assistance</h2>
            <p style='color: #ffffff; font-size: 16px;'>Start a conversation by typing a message below</p>
        </div>
    """, unsafe_allow_html=True)
