import numpy as np
from numpy import linalg

from ..base import Function
from ..utils import null_space
from ..problem import Problem


def equality_elimination(problem):
  """Remove linear equality constraints from a problem

  Change a problem from,

    min_{x} f_{0}(x)
    s.t.    f_{i}(x) <= 0
            Ax = b

  to

    min_{z} f_{0}(Fz + x_hat)
    s.t.    f_{i}(Fz + x_hat) <= 0

  where F is a basis for the null space of A and x_hat is a solution to Ax=b.
  """

  (null, x_hat) = LinearlyUnconstrainedFunction.eliminate(problem.equality_constraint)

  objective = LinearlyUnconstrainedFunction(problem.objective, null=null, x_hat=x_hat)
  inequality_constraints = [
      LinearlyUnconstrainedFunction(constraint, null=null, x_hat=x_hat)
      for constraint in problem.inequality_constraints
  ]

  return Problem(objective, inequality_constraints=inequality_constraints)


def log_barrier(problem, t=1.0):
  """Add log barrier to objective function

  Change a problem from,

    min_{x} f_{0}(x)
    s.t.    f_{i}(x) <= 0
            Ax = b

  to

    min_{z} f_{0}(x) + (1/t) \sum_{i} log(-f_{i}(x))
    s.t.    Ax = b

  The resulting problem is readily solved by interior point methods.
  """
  objective = LogBarrierFunction(problem.objective, problem.inequality_constraints, t=t)
  equality_constraint = objective.equality_constraint
  return Problem(objective, equality_constraint=equality_constraint)


class LinearlyUnconstrainedFunction(Function):
  """Wrap a function to remove equality constraints

  Transform

    f(x) s.t. Ax = b

  into

    f(Fz + x_hat)

  where AFz = 0 for all z and A x_hat = b
  """

  def __init__(self, function, constraint=None, null=None, x_hat=None):

    if null is None or x_hat is None:
      (null, x_hat) = self.eliminate(constraint)

    self.function = function
    self.null = null
    self.x_hat = x_hat

  def eval(self, x):
    null, x_hat, function = self.null, self.x_hat, self.function
    return function(self.recover(x))

  def gradient(self, x):
    null, x_hat, function = self.null, self.x_hat, self.function
    return null.T.dot(function.gradient(self.recover(x)))

  def hessian(self, x):
    null, x_hat, function = self.null, self.x_hat, self.function
    return null.T.dot(function.hessian(self.recover(x))).dot(null)

  def recover(self, z):
    null, x_hat, function = self.null, self.x_hat, self.function
    return null.dot(z) + x_hat

  @staticmethod
  def eliminate(constraint):
      A, b = constraint.A, constraint.b
      null = null_space(A)
      x_hat = linalg.lstsq(A, b)[0]
      return (null, x_hat)


class LogBarrierFunction(Function):
  """Add Log Barrier to function

  Transform a function of the form

    f(x) s.t. f_{i}(x) <= 0

  to

    f(x) + (1/t) \sum_{i} log(-f_{i}(x))
  """
  def __init__(self, function, constraints, t=1.0):
    self.function = function
    self.constraints = constraints
    self.t = t

  def objective(self, x):
    constraints, function, t = self.constraints, self.function, self.t
    evals = np.array( [f(x) for f in self.constraints] )
    evals[evals > 0] = 0  # log(0) = -inf anyways
    return (
        function(x)
        - (1.0/t) * np.sum(np.log(-evals))
    )

  def gradient(self, x):
    constraints, function, t = self.constraints, self.function, self.t
    evals = np.array( [f(x) for f in self.constraints] )
    gradients = np.hstack( [f.gradient(x) for f in self.constraints] )
    evals[evals > 0] = 0
    return (
        function.gradient(x)
        - (1.0/t) * np.sum( (1.0/evals) * gradients, axis=1)
    )

  def hessian(self, x):
    constraints, function, t = self.constraints, self.function, self.t
    evals     = [f(x) for f in self.constraints]
    gradients = [f.gradient(x) for f in self.constraints]
    hessians  = [f.hessian(x) for f in self.constraints]

    result = function.hessian(x)
    for (eval, gradient, hessian) in zip(evals, gradients, hessians):
      result += (1.0 / t) * (
          (1.0 / (eval * eval)) * np.outer(gradient, gradient)
          - (1.0 / eval) * hessian
      )
    return result

