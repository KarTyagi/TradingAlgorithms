import numpy as np
import time

start_time = time.time()

# Initialize parameters
S0 = 100
K = 100
T = 0.17
r = 0.04
N =  10**4
u = 1.0001
d = 1/u
opttype = 'C' # or 'P'

def binomial_tree(S0,K,T,r,N,u,d,opttype='C'):
  # Precompute variables
  dt = T / N
  q = (np.exp(r * dt) - d) / (u - d)
  disc = np.exp(-r * dt)

  # Initialize option value at maturity
  C = S0 * d**(np.arange(N,-1,-1)) * u**(np.arange(0,N+1,1))
  C = np.maximum(C - K, np.zeros(N + 1))

  # Backward induction
  for i in np.arange(N,0,-1):
    C = disc * (q * C[1:i+1] + (1-q) * C[0:i]) #up vector: [1:i+1], down vector: [0:i])

  return C[0]

print(binomial_tree(S0,K,T,r,N,u,d,opttype='C'))
print("Calculation time:", time.time() - start_time)