"""
Stopping criterion
"""

import numpy as np


class MaxIterations(object):

  def __init__(self, n=100):
    self.n = n

  def stopping_criterion(self, iteration, **kwargs):
    return iteration >= self.n


class EllipsoidCriterion(object):

  def __init__(self, epsilon):
    self.epsilon = epsilon

  def stopping_criterion(self, P, x, objective, **kwargs):
    g = objective.gradient(x)
    return np.sqrt(g.dot(P).dot(g)) <= self.epsilon


class NewtonDecrement(object):
  """Check if Newton Decrement is below threshold

  Use the fact that,

    lambda(x) = ( g'H^{-1}g )^{1/2}
              = (dx' H dx)^{1/2}

  g = gradient[f](x)
  H = hessian[f](x)
  dx= H^{-1}g

  bounds f(x) - f(x*)
  """

  def __init__(self, epsilon):
    self.epsilon = epsilon

  def stopping_criterion(self, H, direction, **kwargs):
    return direction.dot(H).dot(direction)/2.0 < self.epsilon
