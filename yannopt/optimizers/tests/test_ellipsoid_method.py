import numpy as np
from numpy.testing import assert_allclose

from yannopt.optimizers import EllipsoidMethod
from yannopt.stopping_criteria import EllipsoidCriterion
from yannopt.functions import Quadratic
from yannopt.problem import minimize


class Optimizer(EllipsoidCriterion, EllipsoidMethod):
  def __init__(self):
    EllipsoidCriterion.__init__(self, 1e-4)
    EllipsoidMethod.__init__(self)


def test_ellipsoid_method():
  A = np.array([[1.0, 0.5, 0.0],
                [0.5, 1.0, 0.5],
                [0.0, 0.5, 1.0]])
  b = [1.0, 2.0, 3.0]
  x0 = np.zeros(3)
  P0 = np.eye(3) * 1e4
  objective = Quadratic(A, b)
  problem = minimize(objective)

  optimizer = Optimizer()

  solution = optimizer.optimize(problem, P0, x0)
  assert_allclose(solution, objective.solution(), atol=1e-2)
