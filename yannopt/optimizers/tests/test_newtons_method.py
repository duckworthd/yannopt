import numpy as np
from numpy.testing import assert_allclose

from yannopt.optimizers import NewtonsMethod, QPNewtonsMethod, LinearConstrainedNewtonsMethod
from yannopt.constraints.base import LinearEquality
from yannopt.constraints.elimination import equality_elimination
from yannopt.learning_rates import BacktrackingLineSearch
from yannopt.stopping_criteria import MaxIterations
from yannopt.functions import Quadratic
from yannopt.problem import Problem


class Optimizer(BacktrackingLineSearch, MaxIterations, NewtonsMethod):
  def __init__(self):
    BacktrackingLineSearch.__init__(self)
    MaxIterations.__init__(self, 1)
    NewtonsMethod.__init__(self)


def test_newtons_method():
  A = np.array([[1.0, 0.5, 0.0],
                [0.5, 1.0, 0.5],
                [0.0, 0.5, 1.0]])
  b = [1.0, 2.0, 3.0]
  x0 = np.zeros(3)
  objective = Quadratic(A, b)
  problem = Problem(objective)

  optimizer = Optimizer()

  solution = optimizer.optimize(problem, x0)
  assert_allclose(solution, objective.solution(), atol=1e-5)


class Optimizer2(QPNewtonsMethod):
  def __init__(self):
    QPNewtonsMethod.__init__(self)


def test_qp_newtons_method():
  Q = np.array([[1.0, 0.5, 0.0],
                [0.5, 1.0, 0.5],
                [0.0, 0.5, 1.0]])
  c = [1.0, 2.0, 3.0]
  objective = Quadratic(Q, c)

  A = np.array([[1.0, 0.0, -1.0],
                [0.0, 1.0,  0.5]])
  b = np.array([0.2, 0.4])
  constraint = LinearEquality(A, b)

  problem = Problem(objective, equality_constraint=constraint)

  optimizer = Optimizer2()
  solution = optimizer.optimize(problem)
  x_star = np.array([-2.48,  1.74, -2.68])
  assert_allclose(solution, x_star, atol=1e-5)


class Optimizer3(BacktrackingLineSearch, MaxIterations, LinearConstrainedNewtonsMethod):
  def __init__(self):
    BacktrackingLineSearch.__init__(self)
    MaxIterations.__init__(self, 1)
    NewtonsMethod.__init__(self)


def test_linear_constrained_newtons_method():
  Q = np.array([[1.0, 0.5, 0.0],
                [0.5, 1.0, 0.5],
                [0.0, 0.5, 1.0]])
  c = [1.0, 2.0, 3.0]
  objective = Quadratic(Q, c)

  A = np.array([[1.0, 0.0, -1.0],
                [0.0, 1.0,  0.5]])
  b = np.array([0.2, 0.4])
  constraint = LinearEquality(A, b)

  problem = Problem(objective, equality_constraint=constraint)

  x = np.linalg.lstsq(A, b)[0]

  optimizer = Optimizer3()
  solution = optimizer.optimize(problem, x)
  x_star = np.array([-2.48,  1.74, -2.68])
  assert_allclose(solution, x_star, atol=1e-5)
