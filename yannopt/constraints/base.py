import numpy as np

from ..base import Constraint


def is_linear(constraint):
  return isinstance(constraint, LinearEquality)


class LinearEquality(Constraint):
  """Represents a constraint of the form Ax=b"""

  def __init__(self, A=None, b=None):
    if A is None and b is None:
      self.A = None
      self.b = None
    else:
      self.A = np.atleast_2d(A)
      self.b = np.atleast_1d(b)

  def is_satisfied(self, x):
    A, b = self.A, self.b
    if A is None and b is None:
      return True
    else:
      return np.all(A.dot(x) - b <= 1e-12)

  def also(self, constraint):
    """Stack this linear equality and one more"""
    return LinearEquality.stack([self, constraint])

  @staticmethod
  def stack(constraints):
    """Stack multiple linear equalities"""
    params = [(c.A, c.b) for c in constraints
              if not (c.A is None or c.b is None)]
    As, bs = zip(*params)
    return LinearEquality(np.vstack(As), np.hstack(bs))


class GeneralInequality(Constraint):
  """Represents f(x) <= 0"""

  def __init__(self, function):
    self.function = function

  def is_satisfied(self, x):
    return self.function(x) <= 0


class Null(Constraint):

  def is_satisfied(self, x):
    return True
