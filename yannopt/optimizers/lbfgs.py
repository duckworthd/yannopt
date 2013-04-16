"""
Limited Memory Broyden-Fletch-Goldfarb-Shannon (L-BFGS)
"""
import numpy as np

from ..base import Optimizer
from ..problem import Solution


class LBFGS(Optimizer):
  """Limited Memory BFGS

  Solves the problem,

    min_{x} f(x)

  for f such that gradient[f](x) is easy to compute

  Parameters
  ----------
  k : int
      length of history
  """
  def __init__(self, k):
    self.k = k

  def optimize(self, objective, x0):
    kwargs = {
        'objective': objective,
    }
    sol = Solution(problem=objective, x0=x0)
    iteration = 0
    x = x0

    rhos, ss, ys = [], [], []

    while True:
      if self.stopping_criterion(iteration=iteration, x=x, **kwargs):
        break
      else:
        # for t = 1,2,...
        #   q = gradient[f](x_t)
        #   for i = t-1...t-m
        #     \alpha_i = \rho_i s_i^T q
        #     q = q - \alpha_i y_i
        #   z = H_t^0 q
        #   for i = t-m...t-1
        #     \beta_i = \rho_i y_i^T z
        #     z = z + s_i (\alpha_i - \beta_i)
        #   x_{t+1} = x_{t} - step_{t} z
        #   s_{t} = x_{t+1} - x_{t}
        #   y_{t} = gradient[f](x_{t+1}) - gradient[f](x_t)
        #   \rho_{t} = 1 / (y_t^T s_t)
        alphas = []

        q = objective.gradient(x)
        for (rho_i, s_i, y_i) in reversed(zip(rhos, ss, ys)):
          alpha_i = rho_i * s_i.dot(q)
          q -= alpha_i * y_i
          alphas.append(alpha_i)
        alphas = list(reversed(alphas))

        z = q
        for (alpha_i, rho_i, s_i, y_i) in zip(alphas, rhos, ss, ys):
          beta_i = rho_i * y_i.dot(z)
          z += (alpha_i - beta_i) * s_i

        direction = -z

        eta = self.learning_rate(iteration=iteration, x=x,
            direction=direction, **kwargs)
        x_new = x + eta * direction

        ss.append(x_new - x)
        ys.append(objective.gradient(x_new) - objective.gradient(x))
        rhos.append(1.0 / (ys[-1].dot(ss[-1])))
        x = x_new

        if len(ss) > self.k:
          ss.pop(0), ys.pop(0), rhos.pop(0)
        if np.isinf(rhos[-1]):
          break

        iteration += 1
        sol.scores.append(objective(x))

    sol.x = x
    return sol
