# Quick Start - Running EMIXbot Tools

## 🚀 Quick Setup

### 1. Start Streamlit App
```bash
cd /Users/ravi/Documents/Kartik/EMIXBOT/TradingAlgorithms
source .env
streamlit run streamlit_app.py
```

The app will be available at: **http://localhost:8501**

### 2. Configure Azure OpenAI
1. Enter your API key in the sidebar
2. Enter deployment name (default: gpt-4)
3. Click anywhere to confirm

---

## 💬 Quick Prompt Examples (Copy & Paste)

### Immediate Results (< 10 seconds)
```
What's the current price of Apple?
```

```
Get TSLA stock metrics
```

```
Check MSFT and GOOGL prices
```

### Medium Analysis (10-30 seconds)
```
Do technical analysis on Tesla
```

```
What's the market sentiment for Apple?
```

```
Analyze risk for AAPL, MSFT, GOOGL portfolio
```

### Full Analysis (30-60 seconds)
```
Generate daily analysis report
```

```
Give me complete analysis for Tesla: price, sentiment, technical
```

```
Analyze technology sector - get sentiment and technical for AAPL, MSFT, GOOGL, NVDA
```

---

## 🎯 Tool Triggering Keywords

Copy any prompt with these keywords to trigger specific tools:

### 📈 Stock Price → Use "price", "stock", "ticker"
```
"What's the current price of AAPL?"
"Get stock metrics for Tesla"
"MSFT stock check"
```

### ⚖️ Portfolio Risk → Use "portfolio", "risk", "volatility"
```
"Analyze portfolio risk for AAPL, GOOGL, MSFT"
"Check volatility of my tech stocks"
"Calculate risk metrics"
```

### 💭 Sentiment → Use "sentiment", "bullish", "bearish", "news"
```
"Market sentiment on Apple?"
"Is Tesla bullish or bearish?"
"What's the news saying about MSFT?"
```

### 📊 Technical → Use "technical", "trend", "moving average", "momentum"
```
"Technical analysis for AAPL"
"Is Tesla in an uptrend?"
"MA20, MA50, MA200 for Microsoft"
```

### 📋 Daily Report → Use "report", "daily", "analysis", "today"
```
"Generate daily report"
"Today's analysis"
"Create full market analysis"
```

---

## 🔄 Common Analysis Flows

### Flow 1: Quick Stock Check (2 min)
```
1. "What's AAPL stock price?"
   ↓ (Gets current metrics)
2. "Technical analysis for Apple"
   ↓ (Shows trend, moving averages)
3. "Market sentiment on AAPL?"
   ↓ (Shows positive/negative ratio)
```

### Flow 2: Portfolio Decision (5 min)
```
1. "I'm considering AAPL, TSLA, NVDA - analyze risk"
   ↓ (Shows risk metrics)
2. "Technical outlook for these three"
   ↓ (Shows trends)
3. "What's the sentiment on Tesla?"
   ↓ (Shows news sentiment)
Decision: Buy/Hold/Sell based on all three
```

### Flow 3: Daily Market Overview (3 min)
```
1. "Generate daily analysis report"
   ↓ (Full report generated)
2. "Which stocks have highest impact?"
   ↓ (S8 scores)
3. "Deep dive on top stock"
   ↓ (Detailed analysis)
```

---

## 📊 What Each Tool Returns

### Stock Price Tool
```
Stock: AAPL
Price: $185.50
Volume: 52,304,320
Trades: 1,234,567
VPT: 42.4
```

### Risk Analysis Tool
```
AAPL: Price $185.50
  MA20: $183, MA50: $180, MA200: $175
TSLA: Price $245
  MA20: $242, MA50: $240, MA200: $235
MSFT: Price $420
  MA20: $415, MA50: $410, MA200: $405
```

### Sentiment Tool
```
Market Sentiment for 'Apple':
Positive: 12/18 (67%)
Negative: 4/18 (22%)
Neutral: 2/18 (11%)
```

### Technical Analysis Tool
```
Technical Analysis for AAPL:
Current Price: $185.50
MA20: $183.20
MA50: $180.75
MA200: $175.40
Momentum Factor: 0.0324
Trend: Strong Uptrend 📈
```

### Daily Report Tool
```
Daily Analysis Report - 2026-01-01
Total Articles Analyzed: 25
Unique Symbols Found: 18

AAPL: Apple posts stronger profits
  Price: $185.50, S8 Score: 0.856
  Sentiment: Positive

TSLA: Tesla faces supply chain issues
  Price: $245.00, S8 Score: -0.234
  Sentiment: Negative

[... more stocks ...]
```

---

## ⚡ Performance Tips

1. **Cache Results** - Same stock analysis repeated = instant results
2. **Batch Analysis** - Analyze multiple stocks in one prompt
3. **Narrow Down** - Start broad, then dig deep
4. **Off-Hours** - Less API lag after market close

---

## 🐛 If Something Goes Wrong

### Error: "Azure OpenAI API key is not configured"
→ Add your API key in the sidebar configuration section

### Error: "No articles found for..."
→ Market may be closed, try: "Get AAPL price" instead

### Error: "Invalid stock symbol"
→ Use standard ticker symbols (AAPL, TSLA, MSFT, etc.)

### Error: "Connection timeout"
→ Check internet connection, try simpler query first

---

## 🎓 Learning Path

**Day 1: Basics**
- Try: "AAPL price"
- Try: "TSLA technical analysis"
- Try: "MSFT sentiment"

**Day 2: Combinations**
- Try: "Analyze AAPL: price, technical, sentiment"
- Try: "Portfolio risk for AAPL, TSLA, MSFT"

**Day 3: Advanced**
- Try: "Daily report"
- Try: "Compare AAPL and TSLA with full analysis"
- Try: "Sector analysis for technology stocks"

---

**Start with any prompt above and explore! The AI will guide you. 🚀**
