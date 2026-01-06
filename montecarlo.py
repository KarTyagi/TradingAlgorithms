import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time

start_time = time.time()

# Stock vars
S = 100
u = 0.07
T = 1
dt = 1/252
N = int(T/dt)
sims = 1000

# Option vars
K = 100

# Data storage arrays
final_prices = []
percent_changes = []
volatilities = []
sharpe = []
all_ma20 = []
C_list = []
sample_prices = []

# Loop to generate simulations
for i in range(sims):

    # Bias s lower using beta distribution
    a, b = 2, 5
    s = np.random.beta(a, b) * (0.4 - 0.05) + 0.05
    prices = [S]
    daily_returns = []

    for _ in range(N):

        shock = np.random.normal(loc=(u * dt), scale=(s*np.sqrt(dt)))
        prices.append(prices[-1] * np.exp(shock))
        daily_returns.append((prices[-1] - prices[-2]) / prices[-2])

    if i < 10:
        sample_prices.append(prices.copy())

    # Price is on the first graph
    #plt.plot(prices, label=f'Sim {i+1} Price')

    # Table data calculations
    final_prices.append(prices[-1])
    percent_changes.append(100 * (prices[-1] - S) / S)
    volatilities.append((np.std(daily_returns) * np.sqrt(252))*100)
    sharpe.append((final_prices[-1] - S) / volatilities[-1] if volatilities[-1] != 0 else 0)

    prices_array = np.array(prices)
    ma20 = np.convolve(prices_array, np.ones(20)/20, mode='valid')
    all_ma20.append(ma20)

    # Premium calculation
    d1 = (np.log(S / K) + (u + 0.5 * s**2) * T) / (s * np.sqrt(T))
    d2 = d1 - s * np.sqrt(T)
    C = S * norm.cdf(d1) - K * np.exp(-u * T) * norm.cdf(d2)
    C_list.append(C)

# Data table
#print("\nData Table:")
#print("\n{:<6} {:>12} {:>12} {:>12} {:>12} {:>12}".format("Sim#", "% Change", "Volatility", "Sharpe", "Premium", "Payoff"))
#print("-" * 75)

#for i in range(sims):
#    print("{:<6} {:>12.2f}% {:>12.4f} {:>12.4f} {:>12.4f} {:>12.4f}".format(
#        i+1, percent_changes[i], volatilities[i], sharpe[i], C_list[i], C_list[i] - max(final_prices[i] - K, 0)
#    ))

# Price graph
plt.figure()
plt.title('Monte Carlo Sim - Sample Price Paths')
plt.xlabel("Days")
plt.ylabel("Price")
for idx, path in enumerate(sample_prices):
    plt.plot(path, label=f'Sim {idx+1} Price')
plt.show()
plt.show()

# Distribution plot
end_time = time.time()
print(f"Simulation Time: {end_time - start_time:.2f} seconds")

plt.figure()
plt.hist(percent_changes, bins=30, edgecolor='black', label='Percent Changes')
plt.title('Distribution of Returns')
plt.xlabel('Percent Change')
plt.ylabel('Frequency')
plt.show()

avg_return = np.mean([C_list[i] - max(final_prices[i] - K, 0) for i in range(sims)])
stdv_returns = np.std([C_list[i] - max(final_prices[i] - K, 0) for i in range(sims)])
print(f"Average Return: {avg_return:.2f}%")
print(f"Standard Deviation of Returns: {stdv_returns:.2f}%")