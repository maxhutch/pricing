import numpy as np
from copy import deepcopy

class distribution:
    pass

def price(deltas, alpha, cis):
  """ Compute the price as a function of cost 
  
  Assumes uniform distribution of costs
  """
  # N is the number of firms
  N = len(deltas)

  # M is the number of cost levels
  if len(cis.shape) == 2:
    M = cis.shape[1]
  else:
    M = 1

  # compute the cost distributions
  costs = []
  for i in range(N):
    # c.x are the cost levels and c.y are the probabilities of those levels
    c = distribution()
    if M == 1:
        c.x = np.array([cis[i],])
    else:
        c.x = cis[i,:] 
    c.y = np.ones((M,1))
    c.y = c.y / np.sum(c.y)
    costs.append(c)

  # Starting guess the prices at the costs
  prices = deepcopy(costs)

  # Transform the price distribution to the exp(delta-alpha p) distribution
  exps = deepcopy(prices)
  for e,d in zip(exps, deltas):
    e.x = np.exp(d - alpha*e.x) 

  # Iterate to solve
  K = 20
  for it in range(K):
    # for each firm
    for p, e, c, d, n in zip(prices, exps, costs, deltas, list(range(N))):
      # first, we compute the distribution of the denominator
      # first, the denom sum has value 0 with probability 1
      old = {0: 1.}
      for j in range(N):
          # the denominator doesn't include the i == j term
          if j == n:
              continue
          new = {}
          # for each old value of the denominator
          for oval, oprob in old.items():
              # for each value of the term
              for l in range(M):
                  # the new value is the sum of the old and term values
                  value = oval + exps[j].x[l]
                  # the probabilty of that value is the product ...
                  prob = oprob * exps[j].y[l]
                  # if we've seen that value before, add probs
                  if value in new:
                      new[value] += prob 
                  else:
                      new[value] = prob 
          old = new
      # check that the sum of the probabilities is 1
      if np.abs(np.sum(np.array(list(old.values()))) - 1) > 0.000001:
          print("Lost norm! {:}".format(np.sum(np.array(list(old.keys())))))
      # for each cost level
      for i in range(M):
          # the expectation is the sum of probs * values
          expect = 0
          for value, prob in old.items():
            expect += (1 + value + e.x[i]) / (1 + value) * prob
          # pricing formula
          new_price = c.x[i] + expect
          # we relax the change by 0.5 to be more robust
          p.x[i] = (new_price - p.x[i]) *.5 + p.x[i]
          # recompute the expoential term
          e.x[i] = np.exp(d - alpha*p.x[i])
  # return the prices
  return [prices[i].x[:] for i in range(N)]

# load data files
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

