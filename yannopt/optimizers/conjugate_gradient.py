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
    iteration = 0
    x = x0
    previous_gradient = 0
    previous_direction = 0
    n_dim = len(x)

    scores = []
    while True:
      if self.stopping_criterion(iteration=iteration, x=x, **kwargs):
        break
      else:
        if iteration % n_dim == 0:
          previous_direction = 0
          previous_gradient  = 0

        gradient = objective.gradient(x)
        if not (np.all(previous_gradient == 0) or np.all(gradient == 0)):
          b = gradient.dot(gradient) / (gradient.dot(previous_gradient))
        else:
          b = 1.0
        direction = -1 * (gradient + b * previous_direction)
        eta = self.learning_rate(iteration=iteration, x=x,
            direction=direction, **kwargs)

        x  += eta * direction

        iteration += 1
        scores.append(objective(x))
        previous_direction = direction
        previous_gradient  = gradient

    return Solution(x=x, scores=scores)
