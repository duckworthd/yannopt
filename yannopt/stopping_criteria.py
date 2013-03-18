"""
Stopping criterion for Gradient-based methods
"""


class MaxIterations(object):

  def __init__(self, n=100):
    self.n = n

  def stopping_criterion(self, iteration, **kwargs):
    return iteration >= self.n
