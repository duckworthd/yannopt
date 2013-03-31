import numpy as np

from yannopt.optimizers import EllipsoidMethod
from yannopt.stopping_criteria import EllipsoidCriterion
from yannopt.testing import check_optimizer
from yannopt.testing import problems


class Optimizer(EllipsoidCriterion, EllipsoidMethod):
  def __init__(self):
    EllipsoidCriterion.__init__(self, 1e-2)
    EllipsoidMethod.__init__(self)


def test_ellipsoid_method():
  solutions = [
      problems.quadratic_program1(),
      problems.logistic_regression(),
  ]
  optimizer = run_ellipsoid_method
  for solution in solutions:
    yield check_optimizer, optimizer, solution


def run_ellipsoid_method(solution):
  n = len(solution.x0)
  P0 = np.eye(n) * 1e4
  x0 = solution.x0
  return Optimizer().optimize(solution.problem, P0, x0)
