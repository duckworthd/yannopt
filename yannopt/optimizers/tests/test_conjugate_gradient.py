import numpy as np
from numpy.testing import assert_allclose

from yannopt.constraints.elimination import equality_elimination
from yannopt.optimizers import ConjugateGradient
from yannopt.learning_rates import DecreasingRate
from yannopt.stopping_criteria import MaxIterations
from yannopt.tests import problems


class Optimizer(DecreasingRate, MaxIterations, ConjugateGradient):
  def __init__(self):
    DecreasingRate.__init__(self)
    MaxIterations.__init__(self, 200)
    ConjugateGradient.__init__(self)


def test_conjugate_gradient():
  """Test on unconstrained QP"""
  optimizer = Optimizer()
  solution  = problems.quadratic_program1()

  solution2 = optimizer.optimize(solution.problem, solution.x0)
  assert_allclose(solution2.x, solution.x, atol=1e-2)


def test_conjugate_gradient2():
  """Test on linearly constrained QP"""
  solution = problems.quadratic_program2()
  problem = equality_elimination(solution.problem)

  optimizer = Optimizer()

  solution2 = optimizer.optimize(problem, np.zeros(1))
  x = problem.objective.recover(solution2.x)

  assert_allclose(x, solution.x, atol=1e-5)
