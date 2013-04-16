"""
Mirror Descent
"""
import numpy as np

from ..base import Optimizer
from ..problem import Solution


class MirrorDescent(Optimizer):
  """Mirror Descent

  Solves the problem,

    min_{x} f(x) + g(x)

  for f and g such that computation of gradient[f](x) and gradient[g^{*}](y) is
  efficient.
  """

  def optimize(self, objective, x0):
    kwargs = {
        'objective': objective,
    }
    sol = Solution(problem=objective, x0=x0)
    iteration = 0
    w = x0
    theta = np.zeros(w.shape)
    u = np.zeros(w.shape)

    f, g = objective.objective.functions
    g_dual = g.conjugate

    while True:
      if self.stopping_criterion(iteration=iteration, x=w, **kwargs):
        break
      else:
        # for t = 1,2,...
        #   z^{t}         = gradient[f](w^{t})
        #   \theta^{t+1}  = \theta^{t} - z^{t}
        #   w^{t+1}       = \argmax{w} <w, (1/t) \theta^{t+1}> - g(w)
        #                 = gradient[g^{*}]( (1/t) \theta^{t+1} )

        t     = iteration
        z     = f.gradient(w)
        theta = (t/(t+1.0)) * theta + (1.0/(t+1.0)) * z
        w     = g_dual.gradient(-theta)

        iteration += 1
        sol.scores.append(objective(w))

    sol.x = w
    return sol
