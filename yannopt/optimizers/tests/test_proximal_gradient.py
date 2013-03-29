from numpy.testing import assert_allclose

from yannopt.optimizers import ProximalGradient, AcceleratedProximalGradient
from yannopt.learning_rates import ProximalBacktrackingLineSearch
from yannopt.stopping_criteria import MaxIterations
from yannopt.tests import problems


class Optimizer(ProximalBacktrackingLineSearch, MaxIterations, ProximalGradient):
  def __init__(self, n_iter):
    ProximalBacktrackingLineSearch.__init__(self)
    MaxIterations.__init__(self, n_iter)
    ProximalGradient.__init__(self)


class Optimizer2(ProximalBacktrackingLineSearch, MaxIterations, AcceleratedProximalGradient):
  def __init__(self, n_iter):
    ProximalBacktrackingLineSearch.__init__(self)
    MaxIterations.__init__(self, n_iter)
    AcceleratedProximalGradient.__init__(self)


def test_proximal_gradient():
  optimizer = Optimizer(100)
  solution  = problems.lasso()

  solution2 = optimizer.optimize(solution.problem, solution.x0)
  assert_allclose(solution2.x, solution.x, atol=1e-2)


def test_accelerated_proximal_gradient():
  optimizer = Optimizer2(100)
  solution  = problems.lasso()

  solution2 = optimizer.optimize(solution.problem, solution.x0)
  assert_allclose(solution2.x, solution.x, atol=1e-2)
