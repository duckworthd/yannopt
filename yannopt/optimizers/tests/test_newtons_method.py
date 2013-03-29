from numpy.testing import assert_allclose

from yannopt.optimizers import NewtonsMethod, QPNewtonsMethod, LinearConstrainedNewtonsMethod
from yannopt.learning_rates import BacktrackingLineSearch
from yannopt.stopping_criteria import MaxIterations
from yannopt.tests import problems


class Optimizer(BacktrackingLineSearch, MaxIterations, NewtonsMethod):
  def __init__(self):
    BacktrackingLineSearch.__init__(self)
    MaxIterations.__init__(self, 1)
    NewtonsMethod.__init__(self)


class Optimizer2(QPNewtonsMethod):
  def __init__(self):
    QPNewtonsMethod.__init__(self)


class Optimizer3(BacktrackingLineSearch, MaxIterations, LinearConstrainedNewtonsMethod):
  def __init__(self):
    BacktrackingLineSearch.__init__(self)
    MaxIterations.__init__(self, 1)
    NewtonsMethod.__init__(self)


def test_newtons_method():
  optimizer = Optimizer()
  solution  = problems.quadratic_program1()

  solution2 = optimizer.optimize(solution.problem, solution.x0)
  assert_allclose(solution2.x, solution.x, atol=1e-5)


def test_qp_newtons_method():
  optimizer = Optimizer2()
  solution  = problems.quadratic_program2()

  solution2 = optimizer.optimize(solution.problem, solution.x0)
  assert_allclose(solution2.x, solution.x, atol=1e-5)


def test_linear_constrained_newtons_method():
  optimizer = Optimizer3()
  solution  = problems.quadratic_program2()

  solution2 = optimizer.optimize(solution.problem, solution.x0)
  assert_allclose(solution2.x, solution.x, atol=1e-2)
