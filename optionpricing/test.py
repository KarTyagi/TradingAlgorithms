import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import norm, binned_statistic_2d
import time

start_time = time.time()

S = 100
u = 0.04
K = 100
sims = 10**5
cf = 1.25
gamma = 0.1
delta = 0.05

# Volatility and DTE grid
s_grid = np.linspace(0.05, 0.4, 30)
T_grid = np.linspace(0.1, 2.0, 30)

avg_profit = np.zeros((len(s_grid), len(T_grid)))

for i, s in enumerate(s_grid):
    for j, T in enumerate(T_grid):
        N = int(T * 252)
        a, b = 2, 5

        shocks = np.random.normal(loc=(u / 252), scale=(s * np.sqrt(1/252)), size=(sims, N))
        log_returns = np.concatenate([np.zeros((sims, 1)), shocks], axis=1)
        log_price_paths = np.cumsum(log_returns, axis=1)
        price_paths = S * np.exp(log_price_paths)
        final_prices = price_paths[:, -1]
        percent_changes = 100 * (final_prices - S) / S

        d1 = (np.log(S / K) + (u + 0.5 * s**2) * T) / (s * np.sqrt(T))
        d2 = d1 - s * np.sqrt(T)
        C = (S * norm.cdf(d1) - K * np.exp(-u * T) * norm.cdf(d2)) * cf + gamma * s**2 + delta * T**2
        seller_profit = C - np.maximum(final_prices - K, 0)
        avg_profit[i, j] = np.mean(seller_profit)

S_mesh, T_mesh = np.meshgrid(s_grid, T_grid, indexing='ij')

# Simulations end
print(f"Simulation Time: {time.time() - start_time:.2f}s")

# Find avg and stdev
avg_return = np.mean(avg_profit)
stdv_returns = np.std(avg_profit)
print(f"Average Seller Return: {avg_return:.2f}")
print(f"Standard Deviation of Seller Returns: {stdv_returns:.2f}")

# plot 3d graph
fig = plt.figure(figsize = (10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(S_mesh, T_mesh, avg_profit, cmap='viridis', edgecolor='none')

ax.set_xlabel('Volatility')
ax.set_ylabel('DTE')
ax.set_zlabel('Profit')
ax.set_title('Profit Surface')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
plt.show()

#plt.figure()
#plt.hist(percent_changes, bins=30, edgecolor='black')
#plt.title('Distribution of Returns')
#plt.xlabel('Percent Change')
#plt.ylabel('Frequency')
#plt.show()