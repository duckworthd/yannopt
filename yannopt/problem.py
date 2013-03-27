
import numpy as np

from .constraints.base import is_linear, LinearEquality
from .base import Function


class Problem(Function):

  def __init__(self, objective, inequality_constraints=[],
               equality_constraint=LinearEquality()):
    self._objective = objective
    self._inequality_constraints = inequality_constraints
    self._equality_constraint = equality_constraint

  def eval(self, x):
    if (self.equality_constraint.is_satisfied(x)
        and all(c.is_satisfied(x) for c in self.inequality_constraints)):
      return self.objective(x)
    else:
      return np.inf

  def gradient(self, x):
    # TODO should this be defined?
    return self.objective.gradient(x)

  def hessian(self, x):
    # TODO should this be defined?
    return self.objective.hessian(x)

  def subject_to(self, constraint):
    if is_linear(constraint):
      # stack equality constraints
      self._equality_constraint = self._equality_constraint.also(constraint)
    else:
      self._inequality_constraints.append(constraint)
    return self

  @property
  def objective(self):
    return self._objective

  @property
  def inequality_constraints(self):
    return self._inequality_constraints

  @property
  def equality_constraint(self):
    return self._equality_constraint


def minimize(function):
  return Problem(function)
