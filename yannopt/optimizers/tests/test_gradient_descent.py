from numpy.testing import assert_allclose

from yannopt.optimizers import SubgradientDescent, AcceleratedGradientDescent
from yannopt.learning_rates import BacktrackingLineSearch
from yannopt.stopping_criteria import MaxIterations
from yannopt.tests import problems


class Optimizer(BacktrackingLineSearch, MaxIterations, SubgradientDescent):
  def __init__(self, n_iter):
    BacktrackingLineSearch.__init__(self, a=0.0)
    MaxIterations.__init__(self, n_iter)
    SubgradientDescent.__init__(self)


class Optimizer2(BacktrackingLineSearch, MaxIterations, AcceleratedGradientDescent):
  def __init__(self, n_iter):
    BacktrackingLineSearch.__init__(self, a=0.5, save_step_size=True)
    MaxIterations.__init__(self, n_iter)
    AcceleratedGradientDescent.__init__(self)


def test_gradient_descent():
  optimizer = Optimizer(50)
  solution  = problems.quadratic_program1()

  solution2 = optimizer.optimize(solution.problem, solution.x0)
  assert_allclose(solution2.x, solution.x, atol=1e-2)


def test_gradient_descent2():
  optimizer = Optimizer(200)
  solution  = problems.lasso()

  solution2 = optimizer.optimize(solution.problem, solution.x0)
  assert_allclose(solution.problem(solution2.x), solution.problem(solution.x), atol=1e-3)


def test_accelerated_gradient_descent():
  optimizer = Optimizer2(200)
  solution  = problems.quadratic_program1()

  solution2 = optimizer.optimize(solution.problem, solution.x0)
  assert_allclose(solution.problem(solution2.x), solution.problem(solution.x), atol=1e-3)
