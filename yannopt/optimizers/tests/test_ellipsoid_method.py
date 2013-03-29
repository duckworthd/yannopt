import numpy as np
from numpy.testing import assert_allclose

from yannopt.optimizers import EllipsoidMethod
from yannopt.stopping_criteria import EllipsoidCriterion
from yannopt.tests import problems


class Optimizer(EllipsoidCriterion, EllipsoidMethod):
  def __init__(self):
    EllipsoidCriterion.__init__(self, 1e-4)
    EllipsoidMethod.__init__(self)


def test_ellipsoid_method():
  P0 = np.eye(3) * 1e4
  solution  = problems.quadratic_program1()
  optimizer = Optimizer()

  solution2 = optimizer.optimize(solution.problem, P0, solution.x0)
  assert_allclose(solution2.x, solution.x, atol=1e-2)
