
import numpy as np

from ..base import Optimizer
from ..problem import Solution


class ADMM(Optimizer):
  """Alternating Direction Method of Multipliers

  Solves the problem,

    min_{x} f(x) + g(x)

  for f and g such that prox_{eta f}(.)  and prox_{eta g}(.) is efficient.
  Technically ADMM can be generalized to solve,

    min_{x,z} f(x) + g(z)
    s.t. Ax + bz = c

  but this form is not implemented here.
  """

  def __init__(self, rho=1.0):
    self.rho = rho

  def optimize(self, objective, x0):
    kwargs = {
        'objective': objective,
    }
    sol = Solution(problem=objective, x0=x0)
    iteration = 0
    x = z = x0
    u = np.zeros(x.shape)

    f, g = objective.objective.functions
    eta = 1.0/self.rho

    while True:
      if self.stopping_criterion(iteration=iteration, x=x, z=z, **kwargs):
        break
      else:
        # x^{k+1} = prox_{eta f}( z^{k}   - u^{k} )
        #         = \argmin_{x} eta f(x) + 0.5 ||x-(z^k-u^k)||_2^2
        #         = \argmin_{x} eta f(x) + 0.5 ||x-z^k+u^k||_2^2
        # z^{k+1} = prox_{eta g}( x^{k+1} + u^{k} )
        #         = \argmin_{z} eta g(z) + 0.5 ||z-(x^{k+1}+u^k)||_2^2
        #         = \argmin_{z} eta g(z) + 0.5 ||x^{k+1}-z+u^k||_2^2
        # u^{k+1} = u^{k} + x^{k+1} - z^{k+1}

        x = f.prox(z-u, eta)
        z = g.prox(x+u, eta)
        u = u + x - z

        iteration += 1
        sol.scores.append(objective(x))

    sol.x = x
    return sol
