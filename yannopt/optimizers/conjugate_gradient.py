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
    previous_gradient = 0
    previous_direction = 0
    n_dim = len(x)

    while True:
      if self.stopping_criterion(iteration=iteration, x=x, **kwargs):
        break
      else:
        if iteration % n_dim == 0:
          previous_direction = previous_gradient = np.zeros(n_dim)

        gradient = objective.gradient(x)
        if float(gradient.dot(previous_gradient)) > 0:
          # fletcher-reeves
          b = gradient.dot(gradient) / (gradient.dot(previous_gradient))
        else:
          b = 0.0
        direction = -1 * (gradient + b * previous_direction)
        eta = self.learning_rate(iteration=iteration, x=x,
            direction=direction, **kwargs)

        x  += eta * direction

        iteration += 1
        sol.scores.append(objective(x))
        previous_direction = direction
        previous_gradient  = gradient

    sol.x = x
    return sol
