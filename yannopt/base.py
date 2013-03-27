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

  def objective(self, x):
    raise NotImplementedError("No objective function implemented")

  def gradient(self, x):
    raise NotImplementedError("No objective function gradient implemented")

  def hessian(self, x):
    raise NotImplementedError("No objective function hessian implemented")


class Constraint(object):

  def is_satisfied(self, x):
    raise NotImplementedError("Not implemented")
