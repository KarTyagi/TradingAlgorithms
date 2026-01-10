# Sentiment-Based Trading Signal Generator Documentation

## Overview
This Python script (`script.py`) is a comprehensive tool for generating trading signals by analyzing sentiment in financial news articles. It integrates web scraping, natural language processing (NLP), financial data APIs, and technical analysis to evaluate the potential market impact of news on S&P 500 stocks. The script processes articles from Yahoo Finance, extracts relevant stock mentions, performs multi-layered sentiment analysis, and computes quantitative metrics to produce actionable insights. Results are stored in a CSV file for historical tracking and displayed in formatted tables.

## Key Functionality
- **News Scraping**: Fetches articles from Yahoo Finance for a specified date.
- **Stock Mention Extraction**: Identifies S&P 500 companies mentioned in article titles.
- **Sentiment Analysis**: Uses aspect-based sentiment analysis (ABSA) and fallback methods (keyword matching and general sentiment) to score news relevance.
- **Financial Data Retrieval**: Pulls real-time stock prices, volumes, and trades; fetches historical closing prices for technical indicators.
- **Metric Calculation**: Computes volume per trade (VPT), moving averages (20/50/200-day), momentum factor (M_f), impact score (S_6), and normalized signal (S_8).
- **Data Output**: Generates tables for console display and appends results to `sentiment_results.csv`.
- **Error Handling**: Includes robust exception handling for API failures and data processing.

## Workflow
1. **Initialization**: Loads sentiment analyzers (from `sentiment.py`) and ABSA pipeline.
2. **Article Fetching**: Scrapes up to 25 articles from Yahoo Finance for the current date.
3. **Processing Loop**:
   - Extracts stock mentions from titles.
   - Fetches full article text.
   - Retrieves stock metrics (price, volume, trades).
   - Performs sentiment analysis with fallbacks.
   - Calculates technical metrics using historical data.
   - Computes impact and signal scores.
4. **Output**: Prints processed results, updates CSV, and offers optional raw data view.
5. **Termination**: Handles no-data scenarios gracefully.

## Key Functions
- `get_fmp_historical_closes(symbol, days=200, api_key)`: Fetches last 200 days of closing prices from Financial Modeling Prep API.
- `calculate_moving_averages(closes, windows=[20, 50, 200])`: Computes simple moving averages for given periods.
- `main()`: Orchestrates the entire process, including scraping, analysis, and output.

## Inputs
- **Date**: Automatically uses current date (`datetime.now().date()`).
- **APIs**: Financial Modeling Prep API key (hardcoded; replace for custom use).
- **External Data**: Yahoo Finance articles, stock data from `stocks.py`, sentiment models from `sentiment.py`.
- **Limits**: Processes up to 25 articles to avoid overload.

## Outputs
- **Console Tables**:
  - Main table: Article #, Date, Symbol, Company, Title, Mapped Score, S_8.
  - Optional raw table: Symbol, Price, Volume, Trades, VPT, AD20/50/200, M_f.
- **CSV File**: `sentiment_results.csv` (appends new runs with separators).
- **Debug Logs**: Prints processing steps and errors for troubleshooting.

## Metrics Explained
- **VPT (Volume per Trade)**: Average volume per trade, formatted for readability.
- **Mapped Score**: Sentiment score (-1 to 1) from ABSA or fallbacks.
- **AD (Average Deviation)**: Price deviation from moving averages (20/50/200-day).
- **M_f (Momentum Factor)**: Weighted average of deviations (50% 20-day, 30% 50-day, 20% 200-day).
- **S_6 (Impact Score)**: Calculated as `10^-4 * (VPT * Price) * (1 + 2 * M_f)` if all components available.
- **S_8 (Normalized Signal)**: Transforms S_6 using `2 * norm.cdf(S_6) - 1` (sigmoid-like normalization to [-1, 1]).
- **Phi and T**: Additional transformations (arctan-based) computed for the first row.

## Dependencies
- **Standard Libraries**: `os`, `datetime`, `requests`.
- **Third-Party**: `pandas` (data handling), `numpy` (calculations), `tabulate` (table formatting), `scipy.stats.norm` (normal CDF).
- **Custom Modules**:
  - `sentiment.py`: Provides `initialize_analyzer`, `initialize_absa`, `analyze_sentiment`, `analyze_aspect_sentiment`.
  - `news.py`: Provides `scrape_yf`, `fetch_article_text`.
  - `stocks.py`: Provides `get_stock_price_and_metrics`, `extract_sp500_mentions`.
- **APIs**: Financial Modeling Prep (free tier; API key required).

Install via pip: `pip install pandas numpy tabulate scipy requests`

## Usage
1. Ensure all dependencies and custom modules are available.
2. Run: `python script.py`
3. Monitor console for progress and results.
4. Check `sentiment_results.csv` for historical data.
5. Respond 'y' to view raw data table if prompted.

## Configuration and Customization
- **API Key**: Update in `get_fmp_historical_closes` for Financial Modeling Prep.
- **Date**: Modify `target_date` for historical analysis.
- **Article Limit**: Adjust slicing in `main()` for more/less processing.
- **Sentiment Keywords**: Expand positive/negative lists for better accuracy.
- **Metrics**: Tweak weights in M_f or S_6 formulas for different strategies.

## Notes
- **Performance**: API calls may introduce delays; cache is used for historical data.
- **Accuracy**: Sentiment analysis relies on models; results are probabilistic.
- **Limitations**: Depends on Yahoo Finance structure (may break with site changes); free API limits apply.
- **Security**: API key is exposed; use environment variables in production.
- **Extensions**: Integrate with trading platforms for automated execution based on S_8 signals.
- **Testing**: Run with sample data to validate; check debug prints for issues.

This script provides a complete pipeline for sentiment-driven trading insights, combining qualitative news analysis with quantitative financial metrics.