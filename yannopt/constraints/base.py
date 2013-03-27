import numpy as np

from ..base import Constraint


def is_linear(constraint):
  return isinstance(constraint, LinearEquality)


class LinearEquality(Constraint):
  """Represents a constraint of the form Ax=b"""

  def __init__(self, A, b):
    self.A = np.atleast_2d(A)
    self.b = np.atleast_1d(b)

  def is_satisfied(self, x):
    A, b = self.A, self.b
    return np.all(A.dot(x) - b <= 1e-12)

  @staticmethod
  def stack(constraints):
    As = [c.A for c in constraints]
    bs = [c.b for b in constraints]
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
