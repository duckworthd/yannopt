"""
(Stochastic) (Sub)Gradient Descent
"""

from ..base import Optimizer


class SubgradientDescent(Optimizer):

  def optimize(self, objective, x0):
    kwargs = {
        'objective': objective,
    }
    iteration = 0
    x = x0

    while True:
      if self.stopping_criterion(iteration=iteration, x=x, **kwargs):
        break
      else:
        direction = -1 * objective.gradient(x)
        eta = self.learning_rate(iteration=iteration, x=x,
            direction=direction, **kwargs)
        x  += eta * direction
        iteration += 1

    return x
