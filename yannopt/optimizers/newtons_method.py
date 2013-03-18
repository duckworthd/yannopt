"""
Newton's Method
"""

import numpy as np

from ..base import Optimizer


class NewtonsMethod(Optimizer):
  """Vanilla Implementation of Newton's 2nd order optimizer"""

  def optimize(self, objective, objective_gradient, objective_hessian, x0):
    kwargs = {
        'objective': objective,
        'objective_gradient': objective_gradient,
        'objective_hessian': objective_hessian
    }
    iteration = 0
    x = x0

    while True:
      if self.stopping_criterion(iteration=iteration, x=x, **kwargs):
        break
      else:
        H = objective_hessian(x)
        g = objective_gradient(x)
        direction = np.linalg.pinv(H).dot(g)
        eta = self.learning_rate(iteration=iteration, x=x,
            direction=direction, **kwargs)

        x  -= eta * direction

        iteration += 1

    return x
