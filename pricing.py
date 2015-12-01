import numpy as np
from copy import deepcopy

class distribution:
    pass

def price(deltas, alpha, cis):
  N = len(deltas)
  if len(cis.shape) == 2:
    M = cis.shape[1]
  else:
    M = 1

  costs = []
  for i in range(N):
    c = distribution()
    if M == 1:
        c.x = np.array([cis[i],])
    else:
        c.x = cis[i,:] 
    c.y = np.ones((M,1))
    c.y = c.y / np.sum(c.y)
    costs.append(c)

  prices = deepcopy(costs)
  exps = deepcopy(prices)
  for e,d in zip(exps, deltas):
    e.x = np.exp(d - alpha*e.x) 

  K = 20
  for it in range(K):
    for p, e, c, d, n in zip(prices, exps, costs, deltas, list(range(N))):
      old = {0: 1.}
      for j in range(N):
          if j == n:
              continue
          new = {}
          for k, v in old.items():
              for l in range(M):
                  if k + exps[j].x[l] in new:
                      new[k + exps[j].x[l]] += v * exps[j].y[l]
                  else:
                      new[k + exps[j].x[l]] = v * exps[j].y[l]
          old = new
      if np.abs(np.sum(np.array(list(old.values()))) - 1) > 0.000001:
          print("Lost norm! {:}".format(np.sum(np.array(list(old.keys())))))
      for i in range(M):
          expect = 0
          for k, v in old.items():
            expect += (1 + k + e.x[i]) / (1 + k) * v
          p.x[i] = (c.x[i] + expect - p.x[i]) *.5 + p.x[i]
          e.x[i] = np.exp(d - alpha*p.x[i])
  return [prices[i].x[:] for i in range(N)]

delta_in = np.loadtxt('./deltas', delimiter=',')
cost_in = np.loadtxt('./cost', delimiter=',')
alpha_in = np.loadtxt('./alphas', delimiter=',')

print("Fixed pricing")
for i in range(9):
  res = price(delta_in[:,i], alpha_in[i], cost_in[:,1])
  print(np.array(res))

print("Variable pricing")
for i in range(9):
  res = price(delta_in[:,i], alpha_in[i], cost_in)
  print(np.array(res))

