from yannopt.optimizers import ConjugateGradient
from yannopt.learning_rates import BacktrackingLineSearch
from yannopt.stopping_criteria import MaxIterations
from yannopt.testing import check_optimizer
from yannopt.testing import problems


class Optimizer(BacktrackingLineSearch, MaxIterations, ConjugateGradient):
  def __init__(self):
    BacktrackingLineSearch.__init__(self, a=0.5, t=100.0)
    MaxIterations.__init__(self, 50)
    ConjugateGradient.__init__(self)


def test_conjugate_gradient():
  solutions = [
      problems.quadratic_program1(),
      problems.logistic_regression(),
  ]
  optimizer = Optimizer()

  for solution in solutions:
    yield check_optimizer, optimizer, solution
