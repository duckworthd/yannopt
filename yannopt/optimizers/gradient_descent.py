"""
(Stochastic) (Sub)Gradient Descent
"""

from ..base import Optimizer


class GradientDescent(Optimizer):

  def optimize(self, objective, objective_gradient, x0):
    kwargs = {
        'objective': objective,
        'objective_gradient': objective_gradient,
    }
    iteration = 0
    x = x0

    while True:
      if self.stopping_criterion(iteration=iteration, x=x, **kwargs):
        break
      else:
        direction = objective_gradient(x)
        eta = self.learning_rate(iteration=iteration, x=x,
            direction=direction, **kwargs)
        x  -= eta * direction
        iteration += 1

    return x
