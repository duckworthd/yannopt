import numpy as np
from numpy.testing import assert_allclose

from yannopt.optimizers import SubgradientDescent
from yannopt.learning_rates import BacktrackingLineSearch
from yannopt.stopping_criteria import MaxIterations
from yannopt.functions import QuadraticProgram


class Optimizer(BacktrackingLineSearch, MaxIterations, SubgradientDescent):
  def __init__(self):
    BacktrackingLineSearch.__init__(self)
    MaxIterations.__init__(self)
    SubgradientDescent.__init__(self)


def test_gradient_descent():
  A = np.array([[1.0, 0.5, 0.0],
                [0.5, 1.0, 0.5],
                [0.0, 0.5, 1.0]])
  b = [1.0, 2.0, 3.0]
  x0 = np.zeros(3)
  qp = QuadraticProgram(A, b)
  optimizer = Optimizer()

  solution = optimizer.optimize(qp, x0)
  assert_allclose(solution, qp.solution(), atol=1e-5)
