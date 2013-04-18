from yannopt.optimizers import SubgradientDescent, AcceleratedGradientDescent
from yannopt.learning_rates import BacktrackingLineSearch
from yannopt.stopping_criteria import MaxIterations
from yannopt.testing import check_optimizer
from yannopt.testing import problems


class Optimizer(BacktrackingLineSearch, MaxIterations, SubgradientDescent):
  def __init__(self, n_iter):
    BacktrackingLineSearch.__init__(self, a=0.5, t=100.0, save_step_size=False)
    MaxIterations.__init__(self, n_iter)
    SubgradientDescent.__init__(self)


class Optimizer2(BacktrackingLineSearch, MaxIterations, AcceleratedGradientDescent):
  def __init__(self, n_iter):
    BacktrackingLineSearch.__init__(self, a=0.5, save_step_size=True)
    MaxIterations.__init__(self, n_iter)
    AcceleratedGradientDescent.__init__(self)


def test_gradient_descent():
  solutions = [
      problems.quadratic_program1(),
      problems.lasso(),
      problems.logistic_regression(),
      problems.l2_penalized_logistic_regression(),
  ]
  for solution in solutions:
    yield check_optimizer, Optimizer(500), solution


def test_accelerated_gradient_descent():
  solutions = [
      problems.quadratic_program1(),
      problems.logistic_regression(),
  ]
  for solution in solutions:
    yield check_optimizer, Optimizer2(200), solution
