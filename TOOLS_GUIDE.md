# EMIXbot Tools Guide - How to Use in Streamlit

## Overview
The EMIXbot Streamlit application integrates all trading analysis tools. The AI automatically detects which tool to use based on your prompt.

---

## Available Tools & Example Prompts

### 1. 📈 Stock Price Lookup
**Purpose:** Get current stock price, volume, and trading metrics

**Example Prompts:**
```
"What's the current price of Apple stock?"
"Get me the stock metrics for TSLA"
"Check MSFT price and trading volume"
"How much is META stock trading at today?"
```

**What it does:**
- Fetches current stock price
- Returns trading volume
- Shows number of trades executed
- Calculates Volume Per Trade (VPT)

---

### 2. ⚖️ Portfolio Risk Calculator
**Purpose:** Analyze risk metrics for a portfolio of stocks

**Example Prompts:**
```
"Analyze the risk for my portfolio with Apple, Tesla, and Microsoft"
"Calculate portfolio risk for AAPL, GOOGL, AMZN"
"What's the risk profile of tech stocks?"
"Give me risk analysis for NVDA and AMD"
```

**What it does:**
- Fetches moving averages (MA20, MA50, MA200)
- Calculates momentum factors
- Identifies trend strength
- Suggests volatility levels

---

### 3. 💭 Market Sentiment Analysis
**Purpose:** Analyze market sentiment based on latest news

**Example Prompts:**
```
"What's the market sentiment on Apple?"
"Analyze sentiment for technology sector"
"Get sentiment for Meta and Amazon stocks"
"How's the market feeling about Tesla today?"
"Check bullish/bearish sentiment for MSFT"
```

**What it does:**
- Scrapes latest financial news
- Analyzes sentiment (positive, negative, neutral)
- Shows sentiment percentages
- Identifies key themes in news

---

### 4. 📊 Technical Analysis
**Purpose:** Perform technical analysis with moving averages and trends

**Example Prompts:**
```
"Do technical analysis on Apple stock"
"What's the technical outlook for Tesla?"
"Analyze MSFT technical indicators"
"Is Google in an uptrend or downtrend?"
"Give me technical setup for NVDA"
```

**What it does:**
- Calculates momentum factor (M_f)
- Shows MA20, MA50, MA200 levels
- Identifies trend direction
- Determines support/resistance levels

---

### 5. 📊 Daily Report Generator
**Purpose:** Generate comprehensive daily analysis report

**Example Prompts:**
```
"Generate daily analysis report"
"Create a report for today's market activity"
"Give me the daily trading report"
"Analyze all stocks from today's news"
"Show me today's impact scores"
```

**What it does:**
- Scrapes all financial news for the day
- Identifies stocks mentioned in news
- Calculates sentiment for each stock
- Computes S6 and S8 scores
- Generates full impact analysis

---

## Understanding the Scores

### VPT (Volume Per Trade)
```
VPT = Trading Volume / Number of Trades
```
Higher VPT indicates larger average trade size.

### Momentum Factor (M_f)
```
M_f = 0.5 × ((Price - MA20) / MA20) + 
      0.3 × ((Price - MA50) / MA50) + 
      0.2 × ((Price - MA200) / MA200)
```
Measures momentum relative to moving averages.

### S6 Score
```
S6 = (10^-4) × (VPT × Price) × (1 + 2×M_f)
```
Combines volume, price, and momentum for impact assessment.

### S8 Score
```
S8 = 2 × norm.cdf(S6) - 1
```
Normalized impact score ranging from -1 to 1.

---

## Advanced Prompts (Multi-step Analysis)

### Comprehensive Stock Analysis
```
"Analyze Apple: get the current price, do technical analysis, 
and check market sentiment"
```

### Portfolio Analysis with Risk
```
"I'm considering a portfolio of AAPL, GOOGL, and MSFT. 
Calculate the risk and give me technical outlook"
```

### Market Comparison
```
"Compare sentiment and technical setup for Tesla and Lucid Motors"
```

### Trending Stocks Report
```
"What are today's most trending stocks and their impact scores?"
```

### Sector Analysis
```
"Analyze the technology sector - check sentiment and identify 
trending stocks with their technical setups"
```

---

## How the AI Chooses Tools

The system automatically detects keywords in your prompt:

| Keywords | Tool Used |
|----------|-----------|
| "price", "stock", "ticker", symbol names | Stock Price Lookup |
| "portfolio", "risk", "volatility" | Portfolio Risk Calculator |
| "sentiment", "bullish", "bearish" | Market Sentiment Analysis |
| "technical", "RSI", "MACD", "trend" | Technical Analysis |
| "report", "daily", "analysis" | Daily Report Generator |

---

## Best Practices

### 1. Be Specific
❌ Bad: "Tell me about stocks"
✅ Good: "What's the technical analysis for Apple?"

### 2. Include Stock Symbols
❌ Bad: "Check tech stocks"
✅ Good: "Analyze AAPL, MSFT, GOOGL"

### 3. Ask Multiple Things
❌ Bad: "Price of Apple" (one metric)
✅ Good: "Get Apple price, sentiment, and technical analysis"

### 4. Specify Time Frames
❌ Bad: "Historical data"
✅ Good: "200-day moving averages for Tesla"

---

## Example Conversation Flow

**User:** "I want to invest in Tesla. Give me complete analysis."

**Assistant:** 
- Uses Stock Price Lookup → Gets current TSLA price: $245.60
- Uses Technical Analysis → Shows MA20: $240, MA50: $235, MA200: $230
  - Indicates: Strong Uptrend 📈
- Uses Market Sentiment → Analyzes recent Tesla news
  - Sentiment: 75% Positive, 15% Negative, 10% Neutral
- Returns comprehensive recommendation based on all factors

---

## Troubleshooting

### "No articles found"
- Market may be closed or no news available
- Try: "Get sentiment for [specific stock]"

### "Invalid stock symbol"
- Ensure proper stock ticker (e.g., AAPL, TSLA)
- Try: "Get price for Apple" (uses symbol extraction)

### "Error in analysis"
- Check Azure API key is configured
- Ensure stable internet connection
- Try with a single stock first

---

## Tips for Maximum Utility

1. **Start Broad, Then Narrow**
   - First: "Daily analysis report"
   - Then: "Deep dive on top 3 stocks"

2. **Combine Analyses**
   - "Check AAPL sentiment AND technical setup"

3. **Track Over Time**
   - "Generate report for 2024-01-01"
   - "Compare with today's report"

4. **Use Natural Language**
   - The AI understands context
   - "Is tech sector bullish?" → Analyzes sentiment

---

## Running in Streamlit

### Starting the App
```bash
source .env
streamlit run streamlit_app.py
```

### Using the Chat Interface
1. Enter your Azure OpenAI API key in sidebar
2. Type your prompt in the chat input
3. Wait for the AI to analyze and respond
4. View tool usage indicator above responses

### Viewing Tool Execution
- Watch for "🔧 [Tool Name]" indicator
- Shows which tool is being used
- Helps understand AI decision-making

---

## API Limits & Performance

- **Rate Limiting:** Financial Modeling Prep API has free tier limits
- **Response Time:** 5-30 seconds per query depending on analysis
- **Data Freshness:** Stock prices update in real-time
- **News:** Daily news articles fetched from Yahoo Finance

---

## Future Enhancements

- [ ] Real-time stock alerts
- [ ] Portfolio backtesting
- [ ] Predictive scoring
- [ ] Multi-date comparisons
- [ ] Export reports to PDF

---

**Happy Trading! 📈**
