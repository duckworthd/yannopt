"""
Skeleton classes
"""

import numpy as np


class Optimizer(object):

  def optimize(self, **kwargs):
    raise NotImplementedError("An optimizer without an optimize() routine? wtf?")

  def learning_rate(self, **kwargs):
    raise NotImplementedError("Mix in with learning rate")

  def stopping_criterion(self, **kwargs):
    raise NotImplementedError("Mix in with Stopping Criterion")


class Function(object):

  def eval(self, x):
    raise NotImplementedError("Function evaluation not implemented")

  def gradient(self, x):
    raise NotImplementedError("Function gradient not implemented")

  def hessian(self, x):
    raise NotImplementedError("Function hessian not implemented")

  def __call__(self, x):
    return self.eval(x)


class Constraint(object):

  def is_satisfied(self, x):
    raise NotImplementedError("Constraint satisfaction not implemented")
