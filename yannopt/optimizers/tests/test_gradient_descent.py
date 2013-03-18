import numpy as np
from numpy.testing import assert_allclose

from yannopt.optimizers import GradientDescent
from yannopt.learning_rates import BacktrackingLineSearch
from yannopt.stopping_criteria import MaxIterations
from yannopt.loss_functions import QuadraticProgram


class Optimizer(BacktrackingLineSearch, MaxIterations, GradientDescent):
  def __init__(self):
    BacktrackingLineSearch.__init__(self)
    MaxIterations.__init__(self)
    GradientDescent.__init__(self)


def test_gradient_descent():
  A = np.array([[1.0, 0.5, 0.0],
                [0.5, 1.0, 0.5],
                [0.0, 0.5, 1.0]])
  b = [1.0, 2.0, 3.0]
  x0 = np.zeros(3)
  qp = QuadraticProgram(A, b)
  optimizer = Optimizer()

  solution = optimizer.optimize(qp.objective, qp.gradient, x0)
  assert_allclose(solution, qp.solution(), atol=1e-5)
