"""
Mirror Descent
"""
import numpy as np

from ..base import Optimizer
from ..problem import Solution
from ..functions import SquaredL2Norm


class MirrorDescent(Optimizer):
  """Mirror Descent

  Solves the problem,

    min_{x} f(x)

  for f such that gradient[f](x) is easy to compute.

  Parameters
  ----------
  mirror_function : Function with Conjugate, optional
      function such that its conjugate's gradient is efficiently computable.
      Used as a regularizer when taking steps in dual space, but does not
      factor into the objective function. If None, then the SquaredL2Norm is
      used, making the algorithm equivalent to Gradient Descent with a step
      size of 1.
  """

  def __init__(self, mirror_function=None):
    self.mirror_function = mirror_function

  def optimize(self, objective, x0):
    kwargs = {
        'objective': objective,
    }
    sol = Solution(problem=objective, x0=x0)
    iteration = 0
    w = x0
    theta = np.zeros(w.shape)
    u = np.zeros(w.shape)
    f = objective

    if self.mirror_function is None:
      g_dual = SquaredL2Norm(n=len(x0))
    else:
      g_dual = self.mirror_function.conjugate

    while True:
      if self.stopping_criterion(iteration=iteration, x=w, **kwargs):
        break
      else:
        # for t = 1,2,...
        #   z^{t}         = gradient[f](w^{t})
        #   \theta^{t+1}  = \theta^{t} - eta^{t} * z^{t}
        #   w^{t+1}       = \argmax{w} <w, \theta^{t+1}> - g(w)
        #                 = gradient[g^{*}]( \theta^{t+1} )

        t     = iteration
        z     = f.gradient(w)
        eta = self.learning_rate(iteration=iteration, x=w,
                                 direction=-z, **kwargs)
        theta -= eta * z
        w     = g_dual.gradient(theta)

        iteration += 1
        sol.scores.append(objective(w))

    sol.x = w
    return sol
