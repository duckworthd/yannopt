from yannopt.optimizers import LBFGS
from yannopt.learning_rates import BacktrackingLineSearch
from yannopt.stopping_criteria import MaxIterations
from yannopt.testing import check_optimizer
from yannopt.testing import problems


class Optimizer(BacktrackingLineSearch, MaxIterations, LBFGS):
  def __init__(self, n_iter):
    BacktrackingLineSearch.__init__(self, a=0.5, t=100.0)
    MaxIterations.__init__(self, n_iter)
    LBFGS.__init__(self, 2)


def test_gradient_descent():
  solutions = [
      problems.quadratic_program1(),
      problems.lasso(),
      problems.logistic_regression(),
      problems.l2_penalized_logistic_regression(),
  ]
  optimizer = Optimizer(20)
  for solution in solutions:
    yield check_optimizer, optimizer, solution
