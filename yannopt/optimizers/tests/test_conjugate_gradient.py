import numpy as np
from numpy.testing import assert_allclose

from yannopt.constraints import LinearEquality, LinearEqualityConstraint
from yannopt.optimizers import ConjugateGradient
from yannopt.learning_rates import DecreasingRate
from yannopt.stopping_criteria import MaxIterations
from yannopt.loss_functions import QuadraticProgram


class Optimizer(DecreasingRate, MaxIterations, ConjugateGradient):
  def __init__(self):
    DecreasingRate.__init__(self)
    MaxIterations.__init__(self, 200)
    ConjugateGradient.__init__(self)


def test_conjugate_gradient():
  """Test on unconstrained QP"""
  A = np.array([[1.0, 0.5, 0.0],
                [0.5, 1.0, 0.5],
                [0.0, 0.5, 1.0]])
  b = [1.0, 2.0, 3.0]
  x0 = np.zeros(3)
  qp = QuadraticProgram(A, b)
  optimizer = Optimizer()

  solution = optimizer.optimize(qp.objective, qp.gradient, x0)
  assert_allclose(solution, qp.solution(), atol=1e-5)


def test_conjugate_gradient2():
  """Test on linearly constrained QP"""
  Q = np.array([[1.0, 0.5, 0.0],
                [0.5, 1.0, 0.5],
                [0.0, 0.5, 1.0]])
  c = [1.0, 2.0, 3.0]
  unconstrained = QuadraticProgram(Q, c)

  A = np.array([[1.0, 0.0, -1.0],
                [0.0, 1.0,  0.5]])
  b = np.array([0.2, 0.4])
  constraint = LinearEquality(A, b)
  constrained = LinearEqualityConstraint([constraint], unconstrained)
  x0 = np.zeros(1)

  optimizer = Optimizer()

  solution = optimizer.optimize(constrained.objective, constrained.gradient, x0)
  solution = constrained.recover(solution)
  x_star = np.array([-2.48,  1.74, -2.68])

  assert_allclose(solution, x_star, atol=1e-5)
