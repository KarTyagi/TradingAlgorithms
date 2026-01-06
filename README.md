# EMIXBOT

## Overview
EMIXBOT is a comprehensive trading analysis platform that combines Monte Carlo simulations for options pricing, sentiment-based trading signal generation, a Streamlit-based user interface, and OpenAI GPT-4o LLM integration for natural language understanding and responses.

The platform consists of several key components:
- **Monte Carlo Simulation**: Models stock price paths and evaluates call options using Black-Scholes model
- **Sentiment Analysis**: Analyzes financial news articles to generate trading signals
- **Streamlit UI**: Interactive web interface for user interaction
- **LLM Integration**: Uses OpenAI GPT-4o to understand user questions and provide natural language answers

## Monte Carlo Simulation for Options Pricing

### Methodology
- **Stock Price Simulation**: Uses geometric Brownian motion (GBM) to simulate daily stock prices over a 1-year period (252 trading days). Volatility is sampled from a beta distribution (biased toward lower values) to introduce conservatism.
- **Options Valuation**: For each simulation, calculates the Black-Scholes call option premium. The payoff is computed as the maximum of (final price - strike price, 0), and the return is the premium minus the payoff.
- **Key Metrics**:
  - Percent change in stock price
  - Annualized volatility
  - Sharpe ratio (return per unit of volatility)
  - 20-day moving average (calculated but not plotted)
  - Option premium and payoff

### Parameters
- **Stock Parameters**:
  - Initial price (`S`): $100
  - Drift (`u`): 7% annual return
  - Time horizon (`T`): 1 year
  - Time step (`dt`): 1/252 (daily)
- **Option Parameters**:
  - Strike price (`K`): $100
- **Simulation Settings**:
  - Number of simulations (`sims`): 100,000
  - Volatility sampling: Beta distribution with shape parameters a=2, b=5, scaled to 5-40%

### Outputs
- Simulation runtime (printed to console)
- Histogram of percent changes in stock prices
- Average return (premium minus payoff) across all simulations

### Use Cases
- Risk Assessment, Premium Calculation, Strategy Evaluation, Sensitivity Analysis, Backtesting

## Sentiment-Based Trading Signal Generator

### Key Functionality
- **News Scraping**: Fetches articles from Yahoo Finance for a specified date.
- **Stock Mention Extraction**: Identifies S&P 500 companies mentioned in article titles.
- **Sentiment Analysis**: Uses aspect-based sentiment analysis (ABSA) and fallback methods to score news relevance.
- **Financial Data Retrieval**: Pulls real-time stock prices, volumes, and trades; fetches historical closing prices for technical indicators.
- **Metric Calculation**: Computes volume per trade (VPT), moving averages (20/50/200-day), momentum factor (M_f), impact score (S_6), and normalized signal (S_8).

### Workflow
1. Article Fetching from Yahoo Finance
2. Stock mention extraction and full article retrieval
3. Sentiment analysis with fallbacks
4. Financial metrics calculation
5. Impact and signal scoring
6. Data output to CSV and console

### Key Metrics
- **VPT (Volume per Trade)**: Average volume per trade
- **Mapped Score**: Sentiment score (-1 to 1)
- **M_f (Momentum Factor)**: Weighted average of price deviations from moving averages
- **S_6 (Impact Score)**: Calculated as `10^-4 * (VPT * Price) * (1 + 2 * M_f)`
- **S_8 (Normalized Signal)**: Normalized to [-1, 1] using sigmoid transformation

## Streamlit User Interface
The platform features an interactive web-based user interface built with Streamlit, providing an intuitive way to interact with the trading analysis tools.

## OpenAI GPT-4o LLM Integration
The system integrates OpenAI's GPT-4o model to understand user questions in natural language and provide intelligent, context-aware responses about trading analysis, market insights, and platform functionality.

## Dependencies
- `numpy`: Numerical computations
- `matplotlib`: Plotting
- `scipy`: Statistical functions
- `pandas`: Data handling
- `tabulate`: Table formatting
- `requests`: HTTP requests
- `streamlit`: Web UI framework
- `langchain-openai`: OpenAI integration
- Custom modules: `sentiment.py`, `news.py`, `stocks.py`, `tools_module.py`

Install via pip: `pip install -r requirements.txt`

## How to Run

### Monte Carlo Simulation
```bash
python montecarlo.py
```

### Sentiment Analysis Script
```bash
python script.py
```

### Streamlit App
```bash
streamlit run streamlit_app.py
```

## Configuration
- Update API keys in relevant files (e.g., Financial Modeling Prep API key)
- Configure OpenAI/Azure settings for LLM integration
- Adjust simulation parameters as needed

## Notes
- Simulations can be computationally intensive; adjust parameters for performance
- API limits apply for financial data services
- Sentiment analysis is probabilistic and should be used as one of many signals
- For production use, secure API keys and consider rate limiting
