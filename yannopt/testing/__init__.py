from yannopt.base import Optimizer


def check_optimizer(optimize, solution, tol=1e-2):
  """Check if an optimizer's solution is near the true solution"""

  # swap Optimizer object with a method that takes a solution
  if isinstance(optimize, Optimizer):
    optimize = default_optimize(optimize)

  # find optimizer's solution and check it against truth
  solution2 = optimize(solution)
  p   = solution.problem( solution.x)
  p2  = solution.problem(solution2.x)
  assert abs(p-p2) < tol, "Gap: %f >= 0.01" % (p2-p)


def default_optimize(optimizer):
  """Wrap an Optimizer object with a function"""
  def f(solution):
    return optimizer.optimize(solution.problem, solution.x0)
  return f
