from numpy.testing import assert_allclose

from yannopt.optimizers import SubgradientDescent
from yannopt.learning_rates import BacktrackingLineSearch
from yannopt.stopping_criteria import MaxIterations
from yannopt.tests import problems


class Optimizer(BacktrackingLineSearch, MaxIterations, SubgradientDescent):
  def __init__(self):
    BacktrackingLineSearch.__init__(self)
    MaxIterations.__init__(self)
    SubgradientDescent.__init__(self)


def test_gradient_descent():
  optimizer = Optimizer()
  solution  = problems.quadratic_program1()

  x = optimizer.optimize(solution.problem, solution.x0)
  assert_allclose(x, solution.x, atol=1e-5)
