from yannopt.optimizers import ConjugateGradient
from yannopt.learning_rates import DecreasingRate
from yannopt.stopping_criteria import MaxIterations
from yannopt.testing import check_optimizer
from yannopt.testing import problems


class Optimizer(DecreasingRate, MaxIterations, ConjugateGradient):
  def __init__(self):
    DecreasingRate.__init__(self)
    MaxIterations.__init__(self, 50)
    ConjugateGradient.__init__(self)


def test_conjugate_gradient():
  solutions = [
      problems.quadratic_program1()
  ]
  optimizer = Optimizer()

  for solution in solutions:
    yield check_optimizer, optimizer, solution
