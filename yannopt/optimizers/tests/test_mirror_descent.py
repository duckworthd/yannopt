from yannopt.optimizers import MirrorDescent
from yannopt.stopping_criteria import MaxIterations
from yannopt.testing import check_optimizer
from yannopt.testing import problems


class Optimizer(MaxIterations, MirrorDescent):
  def __init__(self, n_iter):
    MaxIterations.__init__(self, n_iter)
    MirrorDescent.__init__(self)


def test_mirror_descent():
  solutions = [
      problems.l2_penalized_logistic_regression(),
  ]
  optimizer = Optimizer(300)
  for solution in solutions:
    yield check_optimizer, optimizer, solution
