
import numpy as np

from ..base import Optimizer


class ADMM(Optimizer):
  """Alternating Direction Method of Multipliers

  Solves the problem,

    min_{x} f(x) + g(x)

  for f and g such that prox_{eta f}(.) is efficient.
  """

  def optimize(self, objective, x0):
    kwargs = {
        'objective': objective,
    }
    iteration = 0
    x = z = x0
    u = np.zeros(x.shape)

    f, g = objective.objective.functions

    while True:
      if self.stopping_criterion(iteration=iteration, x=x, **kwargs):
        break
      else:
        # x^{k+1} = prox_{eta f}( z^{k}   - u^{k} )
        # z^{k+1} = prox_{eta g}( x^{k+1} + u^{k} )
        # u^{k+1} = u^{k} + x^{k+1} - z^{k+1}

        eta = 1.0
        x = f.prox(z-u, eta)
        z = g.prox(x+u, eta)
        u = u + eta*(x - z)

        iteration += 1

    return x
