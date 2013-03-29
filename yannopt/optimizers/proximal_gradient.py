"""
Proximal Gradient Descent
"""

from ..base import Optimizer
from ..problem import Solution


class ProximalGradient(Optimizer):
  """Proximal Gradient Descent

  Solves the problem,

    min_{x} f(x) + g(x)

  for f differentiable and g such that calculating,

    prox_{eta g}(x) = argmin_{y} g(y) + (1/2 eta) ||y-x||_2^2

  is efficient.

  Parameters
  ----------
  objective : Problem
      Problem with no constraints. objective.objective is assumed to implement
      Separable and consist of 2 functions. objective.objective.functions[0] is
      assumed differentiable and objective.objective.functions[1] implements
      Prox.
  x0 : array
      starting value for x
  """

  def optimize(self, objective, x0):
    kwargs = {
        'objective': objective,
    }
    iteration = 0
    x = x0
    f, g = objective.objective.functions

    while True:
      if self.stopping_criterion(iteration=iteration, x=x, **kwargs):
        break
      else:
        # x^{k+1} = prox_{eta^{k} g}( x^{k} - eta^{k} gradient[f](x^{k}) )
        direction = -1 * f.gradient(x)
        eta = self.learning_rate(iteration=iteration, x=x,
            direction=direction, grad_function=f, prox_function=g, **kwargs)

        x = g.prox(x + eta * direction, eta)

        iteration += 1

        #print '%d | %f' % (iteration, objective(x))

    return Solution(x=x)


class AcceleratedProximalGradient(Optimizer):
  """Accelerated Proximal Gradient Descent

  Proximal Gradient Descent + Nesterov's Optimal Gradient Method. Solves
  problems of the same form as ProximalGradient.
  """

  def optimize(self, objective, x0):
    kwargs = {
        'objective': objective,
    }
    iteration = 0
    x = x_best = x0
    y = x_prev = 0

    best_score = objective(x)

    f, g = objective.objective.functions

    while True:
      if self.stopping_criterion(iteration=iteration, x=x, **kwargs):
        break
      else:
        # for k = 0,1,...
        #   y^{k+1} = x^{k} + w^{k} ( x^{k} - x^{k-1} )
        #   x^{k+1} = prox_{eta^{k} g}( y^{k+1} - eta^{k} gradient[f](y^{k+1}) )

        w = iteration / (iteration + 3.0)
        y = x + w * (x - x_prev)
        direction = -1 * f.gradient(y)
        eta = self.learning_rate(iteration=iteration, x=y,
            direction=direction, grad_function=f, prox_function=g, **kwargs)

        x_prev = x
        x = g.prox(y + eta * direction, eta)

        iteration += 1

        score = objective(x)
        if score < best_score:
          best_score = score
          x_best = x

        #print '%d | %f' % (iteration, objective(x))

    return Solution(x=x_best)
