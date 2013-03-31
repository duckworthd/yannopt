from yannopt.optimizers import NewtonsMethod
from yannopt.learning_rates import BacktrackingLineSearch
from yannopt.stopping_criteria import MaxIterations
from yannopt.testing import check_optimizer
from yannopt.testing import problems


class Optimizer(BacktrackingLineSearch, MaxIterations, NewtonsMethod):
  def __init__(self):
    BacktrackingLineSearch.__init__(self)
    MaxIterations.__init__(self, 1)
    NewtonsMethod.__init__(self)


def test_newtons_method():
  solutions = [
      problems.quadratic_program1(),
      problems.quadratic_program2(),
  ]
  optimizer = Optimizer()

  for solution in solutions:
    yield check_optimizer, optimizer, solution
