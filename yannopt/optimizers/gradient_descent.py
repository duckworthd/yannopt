"""
(Stochastic) (Sub)Gradient Descent
"""

from ..base import Optimizer
from ..problem import Solution


class SubgradientDescent(Optimizer):

  def optimize(self, objective, x0):
    kwargs = {
        'objective': objective,
    }
    sol = Solution(problem=objective, x0=x0)
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
        sol.scores.append(objective(x))

    sol.x = x
    return sol


class AcceleratedGradientDescent(Optimizer):
  """Nesterov's Accelerated Gradient Descent

  Note: this method is very sensitive to step size! If using Backtracking Line
  Search, make sure to set a = 0.5 or this algorithm will diverge. The reason
  is that the exit condition for Backtracking Line Search is,

      f(x-tg) < f(x) - (t/2)||g||^2
              = f(x) + g'((x-tg)-x) + (1/2t)||(x-tg)-x||^2

  if t <= 1/L then this is always true as,

      f(y) < f(x) + g'(y-x) + (L/2)||y-x||^2

  for all x and y by the assumption that f is L-Lipschitz. Thus, replacing
  (1/2) with another constant BLS fails to properly approximate the Lipschitz
  constant.
  """

  def optimize(self, objective, x0):
    kwargs = {
        'objective': objective,
    }
    sol = Solution(problem=objective, x0=x0)
    iteration = 0
    x = x_best = x0
    y = x_prev = 0

    best_score = objective(x)

    while True:
      if self.stopping_criterion(iteration=iteration, x=x, **kwargs):
        break
      else:
        # for k = 1,2,...
        #   x^{k} = y^{k} - eta^{k} gradient[f](y^{k})
        #   y^{k} = y^{k} - (k-1)/(k+2) (x^{k} - x^{k-1})
        w = iteration / (iteration + 3.0)
        y = x + w * (x - x_prev)

        direction = -1 * objective.gradient(y)
        eta = self.learning_rate(iteration=iteration, x=y,
            direction=direction, **kwargs)

        x_prev = x
        x = y + eta * direction

        iteration += 1

        score = objective(x)
        if score < best_score:
          best_score = score
          x_best = x

        #print '%d | %f' % (iteration, score)
        sol.scores.append(score)

    sol.x = x_best
    return sol
