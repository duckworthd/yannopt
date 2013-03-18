"""
Functions for dealing with constraints
"""
import numpy as np
from numpy import linalg

from .base import Function
from .utils import null_space


class LinearEquality(Function):
  """Add linear constraints to an unconstrained objective

  Reduce a linear equality constrained problem in R^n to an unconstrained
  problem in R^{null} where R^{null} is the dimensionality of the null space of
  the linear constraints.
  """

  def __init__(self, A, b, function):
    self.A = np.atleast_2d(A)
    self.b = np.atleast_1d(b)
    self.function = function
    self.x_hat = linalg.solve(A, b)
    self.null = null_space(A)

  def objective(self, x):
    null, x_hat, function = self.null, self.x_hat, self.function
    return function.objective(null.dot(x) + x_hat)

  def gradient(self, x):
    null, x_hat, function = self.null, self.x_hat, self.function
    return null.T.dot(function.gradient(null.dot(x) + x_hat))

  def hessian(self, x):
    raise NotImplementedError("TODO")


class Inequalities(Function):
  """Add inequality constraints to a function.

  Uses the log-barrier trick to add smoothness near constraints.
  """
  def __init__(self, constraints, function, t=1.0):
    self.constraints = constraints
    self.function = function
    self.t = t

  def objective(self, x):
    constraints, function, t = self.constraints, self.function, self.t
    evals = np.array( [f.objective(x) for f in self.constraints] )
    evals[evals > 0] = 0
    return (
        function.objective(x)
        - (1.0/t) * np.sum(np.log(-evals))
    )

  def gradient(self, x):
    constraints, function, t = self.constraints, self.function, self.t
    evals = np.array( [f.objective(x) for f in self.constraints] )
    gradients = np.hstack( [f.gradient(x) for f in self.constraints] )
    evals[evals > 0] = 0
    return (
        function.gradient(x)
        - (1.0/t) * np.sum( (1.0/evals) * gradients, axis=1)
    )

  def hessian(self, x):
    constraints, function, t = self.constraints, self.function, self.t
    evals     = [f.objective(x) for f in self.constraints]
    gradients = [f.gradient(x) for f in self.constraints]
    hessians  = [f.hessian(x) for f in self.constraints]
    result = function.gradient(x)
    for (eval, gradient, hessian) in zip(evals, gradients, hessians):
      result += (1.0 / t) * (
          (1.0 / (eval * eval)) * np.outer(gradient, gradient)
          - (1.0 / eval) * hessian
      )
    return result
