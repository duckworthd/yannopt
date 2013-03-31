"""
Stopping criterion
"""

import numpy as np


class StoppingCriteria(object):

  def __call__(self, *args, **kwargs):
    return self.stopping_criterion(*args, **kwargs)


class MaxIterations(object):

  def __init__(self, *args, **kwargs):
    self.stopping_criterion = _MaxIterations(*args, **kwargs)


class _MaxIterations(StoppingCriteria):

  def __init__(self, n=100):
    self.n = n

  def stopping_criterion(self, iteration, **kwargs):
    return iteration >= self.n


class EllipsoidCriterion(object):

  def __init__(self, *args, **kwargs):
    self.stopping_criterion = _EllipsoidCriterion(*args, **kwargs)


class _EllipsoidCriterion(StoppingCriteria):

  def __init__(self, epsilon):
    self.epsilon = epsilon

  def stopping_criterion(self, P, x, objective, **kwargs):
    g = objective.gradient(x)
    return np.sqrt(g.dot(P).dot(g)) <= self.epsilon


class NewtonDecrement(object):

  def __init__(self, *args, **kwargs):
    self.stopping_criterion = _NewtonDecrement(*args, **kwargs)


class _NewtonDecrement(StoppingCriteria):
  """Check if Newton Decrement is below threshold

  Use the fact that,

    f(x) - f(x*) <= lambda(x)
                  = ( g'H^{-1}g )^{1/2}
                  = (dx' H dx)^{1/2}

  where g = gradient[f](x)
        H = hessian[f](x)
        dx= H^{-1}g
  """

  def __init__(self, epsilon):
    self.epsilon = epsilon

  def stopping_criterion(self, H, direction, **kwargs):
    return direction.dot(H).dot(direction)/2.0 < self.epsilon


class GradientNorm(object):

  def __init__(self, *args, **kwargs):
    self.stopping_criterion = _GradientNorm(*args, **kwargs)


class _GradientNorm(StoppingCriteria):
  """Check if the L2 norm of the gradient is small

  Use the fact that,

    f(x) - f(x*) <= (1/2m)||g||_2^2

  where g = gradient[f](x)
        m = strong convexity parameter of f
  """

  def __init__(self, epsilon):
    self.epsilon = epsilon

  def stopping_criterion(self, x, objective, **kwargs):
    g = objective.gradient(x)
    return np.linalg.norm(g) <= self.epsilon
