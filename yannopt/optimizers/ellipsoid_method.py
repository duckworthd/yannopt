"""
Ellipsoid cutting-plane method
"""

import numpy as np

from ..base import Optimizer


class EllipsoidMethod(Optimizer):

  def optimize(self, objective, ellipsoid0, x0):
    assert (objective.equality_constraint.A is None,
            "This method only handles unconstrained problems")
    kwargs = {
        'objective': objective,
    }
    iteration = 0
    P = ellipsoid0
    x = x0
    n = len(x)

    while True:
      if self.stopping_criterion(iteration=iteration, x=x, P=P, **kwargs):
        break
      else:
        # use gradient of a violated constraint (if any), else use the gradient
        # of the objective.
        g = objective.gradient(x)
        for constraint in objective.inequality_constraints:
          if constraint(x) > 0:
            g = constraint.gradient(x)
            break

        g_bar = (1.0/np.sqrt(g.dot(P).dot(g))) * g

        assert (g.dot(P).dot(g) >= 0,
                'Numerical failure due to subtraction of 2 PSD matrices')

        x = x - (1.0/(n+1)) * P.dot(g_bar)
        Pg_bar = P.dot(g_bar)
        P = (n*n / (n*n-1)) * (P - (2.0/(n+1)) * np.outer(Pg_bar, Pg_bar))

        iteration += 1

    return x
