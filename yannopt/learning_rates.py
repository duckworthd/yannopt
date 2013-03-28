"""
Learning Rates for gradient-based methods
"""
import numpy as np


class DecreasingRate(object):
  """Learning rate of the form a / (iter + b)^p"""
  def __init__(self, a=1.0, b=1.0, p=0.5):
    self.a = a
    self.b = b
    self.p = p

  def learning_rate(self, iteration, **kwargs):
    return self.a / ((iteration + self.b) ** self.p)


class BacktrackingLineSearch(object):
  def __init__(self, a=0.1, b=0.9, t0=1e-12):
    self.a = a
    self.b = b
    self.t0 = t0

  def learning_rate(self, x, direction, objective, **kwargs):
    t = 1.0
    score = objective(x)
    gradient = objective.gradient(x)
    while True:
      new_score =  objective(x + t*direction)
      difference = self.a * gradient.dot(direction)
      assert difference <= 1e-12, "direction is not a direction of ascent!"
      if new_score < score + difference:
        return t
      elif t < self.t0:
        # TODO should throw an exception or something?
        print 'WARNING: backtracking line search failed'
        return 0.0
      else:
        t *= self.b


class AdaptiveGradient(object):
  """Adaptive Gradient-based per-feature learning rate"""
  def __init__(self, multiplier=1.0, smoothing=0.1):
    self.weights = smoothing
    self.multiplier = multiplier

  def learning_rate(self, x, direction, **kwargs):
    self.weights += direction ** 2
    return self.multiplier / np.sqrt(self.weights)


class ProximalBacktrackingLineSearch(object):
  """Backtracking search for Proximal methods"""
  def __init__(self, b=0.9):
    self.b = b

  def learning_rate(self, x, direction, prox_function, **kwargs):
    t = 1.0
    f = prox_function

    # make quadratic approximation to f at x while scaling the quadratic term
    # down by (1/l)
    c = f(x)
    g = f.gradient(x)
    f2 = lambda y, l: c + g.dot(y-x) + (0.5 / l) * (np.linalg.norm(y-x) ** 2)

    while True:
      z = f.prox(x + t * direction, t)
      if f(z) <= f2(z, t):
        return t
      t = self.b * t
