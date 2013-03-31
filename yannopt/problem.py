
import numpy as np

from .constraints.base import is_linear, LinearEquality
from .base import Function


def minimize(function):
  """Create a problem

  Parameters
  ----------
  problem : Function
      objective function to be minimized
  """
  return Problem(function)


class Problem(Function):
  """An optimization problem

  An optimization problem of the form

    min_{x} f_{0}(x)
    s.t.    f_{i)(x) <= 0
            Ax = b
  """

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


class Solution(object):
  """Solution to an optimization problem

  Parameters
  ----------
  x : array
      primal variable
  v : array
      dual variable
  problem : Problem
      original problem definition
  x0 : array
      initial primal variable
  v0 : array
      initial dual variable
  duality_gap : float
      upper bound on difference between problem(x) and problem(x*)
  scores : [float]
      problem(x_t) for each iteration t
  """

  def __init__(self, name=None, x=None, v=None, problem=None, x0=None, v0=None,
               duality_gap=None, scores=[]):
    self.name = name
    self.x = x
    self.v = v
    self.problem = problem
    self.x0 = x0
    self.v0 = v0
    self.duality_gap = duality_gap
    self.scores = scores

  def __str__(self):
    return self.name

  def __repr__(self):
    return str(self)
