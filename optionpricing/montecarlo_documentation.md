# Monte Carlo Simulation for Options Pricing and Valuation

## Overview
This Python script (`montecarlo.py`) implements a Monte Carlo simulation to model stock price paths and evaluate call options using the Black-Scholes model. It generates thousands of simulated price trajectories to assess risk, volatility, and option premiums, providing statistical insights into potential returns and payoffs.

## Methodology
- **Stock Price Simulation**: Uses geometric Brownian motion (GBM) to simulate daily stock prices over a 1-year period (252 trading days). Volatility is sampled from a beta distribution (biased toward lower values) to introduce conservatism.
- **Options Valuation**: For each simulation, calculates the Black-Scholes call option premium. The payoff is computed as the maximum of (final price - strike price, 0), and the return is the premium minus the payoff.
- **Key Metrics**:
  - Percent change in stock price
  - Annualized volatility
  - Sharpe ratio (return per unit of volatility)
  - 20-day moving average (calculated but not plotted)
  - Option premium and payoff

## Parameters
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

## Outputs
- Simulation runtime (printed to console)
- Histogram of percent changes in stock prices
- Average return (premium minus payoff) across all simulations
- (Commented-out: Data table with per-simulation metrics and additional plots for price paths and moving averages)

## Use Cases in Options Pricing and Valuation
- **Risk Assessment**: Quantifies potential option payoffs under various volatility scenarios, helping traders evaluate downside risk.
- **Premium Calculation**: Estimates fair option prices using Black-Scholes, adjusted for simulated paths to account for path dependency.
- **Strategy Evaluation**: Analyzes returns to inform hedging or speculative strategies, such as buying calls on undervalued stocks.
- **Sensitivity Analysis**: Tests how changes in volatility (via beta distribution) affect premiums and Sharpe ratios, useful for volatility trading.
- **Backtesting**: Provides synthetic data for testing automated trading algorithms in robo-advisory systems.

## Dependencies
- `numpy`: For numerical computations and random sampling
- `matplotlib`: For plotting distributions
- `scipy.stats`: For normal distribution (Black-Scholes) and beta distribution
- `time`: For performance timing

Install via pip: `pip install numpy matplotlib scipy`

## How to Run
1. Ensure Python 3.x is installed with required dependencies.
2. Execute the script: `python montecarlo.py`
3. View console output for runtime and average return.
4. The histogram plot will display automatically.

## Notes
- Simulations are computationally intensive; reduce `sims` for faster runs.
- Volatility bias (beta distribution) models conservative market conditions; adjust parameters for different scenarios.
- For production use, integrate with real market data and consider parallel processing for larger simulations.