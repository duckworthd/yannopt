from yannopt.optimizers import NewtonsMethod
from yannopt.learning_rates import BacktrackingLineSearch
from yannopt.stopping_criteria import NewtonDecrement
from yannopt.testing import check_optimizer
from yannopt.testing import problems


class Optimizer(BacktrackingLineSearch, NewtonDecrement, NewtonsMethod):
  def __init__(self):
    BacktrackingLineSearch.__init__(self)
    NewtonDecrement.__init__(self, epsilon=1e-2)
    NewtonsMethod.__init__(self)


def test_newtons_method():
  solutions = [
      problems.quadratic_program1(),
      problems.quadratic_program2(),
      problems.logistic_regression()
  ]
  optimizer = Optimizer()

  for solution in solutions:
    yield check_optimizer, optimizer, solution
