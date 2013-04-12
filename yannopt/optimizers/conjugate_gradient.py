"""
Conjugate Gradient Descent
"""

import numpy as np

from ..base import Optimizer
from ..problem import Solution


class ConjugateGradient(Optimizer):

  def optimize(self, objective, x0):
    kwargs = {
        'objective': objective,
    }
    sol = Solution(problem=objective, x0=x0)

    iteration = 0
    x = x0
    previous_gradient = np.zeros(x.shape)
    previous_direction = np.zeros(x.shape)
    n_dim = len(x)

    while True:
      if self.stopping_criterion(iteration=iteration, x=x, **kwargs):
        break
      else:
        gradient = objective.gradient(x)
        if float(gradient.dot(previous_gradient)) > 0:
          # fletcher-reeves
          b = gradient.dot(gradient) / (previous_gradient.dot(previous_gradient))
          b = max(b, 0)
        else:
          b = 0.0
        direction = -gradient + b * previous_direction
        eta = self.learning_rate(iteration=iteration, x=x,
            direction=direction, **kwargs)

        x  += eta * direction

        iteration += 1
        sol.scores.append(objective(x))
        previous_direction = direction
        previous_gradient  = gradient

    sol.x = x
    return sol
