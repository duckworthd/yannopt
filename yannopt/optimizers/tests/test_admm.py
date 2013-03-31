from yannopt.optimizers import ADMM
from yannopt.stopping_criteria import MaxIterations
from yannopt.testing import check_optimizer
from yannopt.testing import problems


class Optimizer(MaxIterations, ADMM):
  def __init__(self):
    MaxIterations.__init__(self, 50)
    ADMM.__init__(self)


def test_admm():
  solutions = [
      problems.lasso()
  ]
  optimizer = Optimizer()

  for solution in solutions:
    yield check_optimizer, optimizer, solution
