from yannopt.optimizers import ProximalGradient, AcceleratedProximalGradient
from yannopt.learning_rates import ProximalBacktrackingLineSearch
from yannopt.stopping_criteria import MaxIterations
from yannopt.testing import check_optimizer
from yannopt.testing import problems


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
  solutions = [
      problems.lasso()
  ]
  optimizer = Optimizer(100)

  for solution in solutions:
    yield check_optimizer, optimizer, solution


def test_accelerated_proximal_gradient():
  solutions = [
      problems.lasso()
  ]
  optimizer = Optimizer2(100)

  for solution in solutions:
    yield check_optimizer, optimizer, solution
