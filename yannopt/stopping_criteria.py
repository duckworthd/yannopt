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
