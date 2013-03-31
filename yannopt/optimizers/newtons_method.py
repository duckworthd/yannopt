"""
Newton's Method
"""

import numpy as np

from ..base import Optimizer
from ..constraints.base import is_linear
from ..problem import Solution


class NewtonsMethod(Optimizer):
  """Newton's Method with (or without) equality constraints

  min_{x} f_{0}(x)
  s.t.    Ax = b
  """

  def optimize(self, objective, x0):
    assert len(objective.inequality_constraints) == 0, "This method cannot handle inequality constraints"
    kwargs = {
        'objective': objective,
    }
    sol = Solution(problem=objective, x0=x0)
    iteration = 0
    x = x0

    while True:
      if self.stopping_criterion(iteration=iteration, x=x, **kwargs):
        break
      elif is_linear(objective.equality_constraint):
        ### CONSTRAINED ###
        # solve the KKT conditions matrix for the following quadratic
        # approximation to the original problem
        #
        # min_{y} f(x) + g'y + (1/2)y'Hy
        # s.t.    A(x + y) = b
        #
        # where g = gradient[f](x)
        #       H =  hessian[f](x)
        #
        # [ H A' ] [ y ] = [-g ]
        # [ A 0  ] [ v ] = [ 0 ]
        #
        # y is then a search direction 
        A, b = objective.equality_constraint.A, objective.equality_constraint.b
        H = objective.hessian(x)
        g = objective.gradient(x)

        n_v, n_x = A.shape

        M = np.zeros((n_x + n_v, n_x + n_v))
        M[0:n_x, 0:n_x]           = H
        M[n_x:n_x+n_v, 0:n_x]     = A
        M[0:n_x, n_x:n_x+n_v]     = A.T

        r = np.hstack([-g, np.zeros(b.shape)])
        y = np.linalg.lstsq(M, r)[0]
        direction = y[0:n_x]
      else:
        ### UNCONSTRAINED ###
        # Solve the following approximation to the original problem
        #
        # min_{y} f(x) + g'y + (1/2) y'Hy
        #
        # where g = gradient[f](x)
        #       H =  hessian[f](x)
        H = objective.hessian(x)
        g = objective.gradient(x)
        direction = np.linalg.lstsq(H, -g)[0]

      eta = self.learning_rate(iteration=iteration, x=x,
          direction=direction, **kwargs)

      x  += eta * direction

      iteration += 1
      sol.scores.append(objective(x))

    sol.x = x
    return sol
