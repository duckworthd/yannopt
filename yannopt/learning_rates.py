"""
Learning Rates for gradient-based methods
"""
import numpy as np


class LearningRate(object):
  def __call__(self, *args, **kwargs):
    return self.learning_rate(*args, **kwargs)


class DecreasingRate(object):
  def __init__(self, *args, **kwargs):
    self.learning_rate = _DecreasingRate(*args, **kwargs)


class _DecreasingRate(LearningRate):
  """Learning rate of the form a / (iter + b)^p"""
  def __init__(self, a=1.0, b=1.0, p=0.5):
    self.a = a
    self.b = b
    self.p = p

  def learning_rate(self, iteration, **kwargs):
    return self.a / ((iteration + self.b) ** self.p)


class BacktrackingLineSearch(object):
  def __init__(self, *args, **kwargs):
    self.learning_rate = _BacktrackingLineSearch(*args, **kwargs)


class _BacktrackingLineSearch(LearningRate):
  def __init__(self, a=0.1, b=0.9, t0=1e-12, t=1.0, save_step_size=False):
    self.a = a
    self.b = b
    self.t0 = t0
    self.t = t
    self.save_step_size = save_step_size

  def learning_rate(self, x, direction, objective, **kwargs):
    t = self.t

    score = objective(x)
    gradient = objective.gradient(x)
    while True:
      new_score =  objective(x + t*direction)
      difference = self.a * t * gradient.dot(direction)
      assert difference <= 1e-12, "direction is not a direction of descent!"
      if new_score < score + difference:
        if self.save_step_size:
          self.t = t
        return t
      elif t < self.t0:
        # TODO should throw an exception or something?
        print 'WARNING: backtracking line search failed'
        return 0.0
      else:
        t *= self.b


class AdaptiveGradient(object):
  def __init__(self, *args, **kwargs):
    self.learning_rate = _AdaptiveGradient(*args, **kwargs)


class _AdaptiveGradient(LearningRate):
  """Adaptive Gradient-based per-feature learning rate"""
  def __init__(self, multiplier=1.0, smoothing=0.1):
    self.weights = smoothing
    self.multiplier = multiplier

  def learning_rate(self, x, direction, **kwargs):
    self.weights += direction ** 2
    return self.multiplier / np.sqrt(self.weights)


class ProximalBacktrackingLineSearch(object):
  def __init__(self, *args, **kwargs):
    self.learning_rate = _ProximalBacktrackingLineSearch(*args, **kwargs)


class _ProximalBacktrackingLineSearch(LearningRate):
  """Backtracking search for Proximal methods"""
  def __init__(self, b=0.9, t=1.0, save_step_size=False):
    self.b = b
    self.t = t
    self.save_step_size = save_step_size

  def learning_rate(self, x, direction, grad_function, prox_function, **kwargs):
    t = self.t

    # make quadratic approximation to the differentiable function  at x while
    # scaling the quadratic term down until 1/l >= its Lipschitz constant
    c = grad_function(x)
    g = grad_function.gradient(x)
    grad_function2 = lambda y, l: c + g.dot(y-x) + (0.5 / l) * (np.linalg.norm(y-x) ** 2)

    while True:
      z = prox_function.prox(x + t * direction, t)
      if grad_function(z) <= grad_function2(z, t):
        if self.save_step_size:
          self.t = t
        return t
      t = self.b * t


class PolyakStepSize(object):
  def __init__(self, *args, **kwargs):
    self.learning_rate = _PolyakStepSize(*args, **kwargs)


class _PolyakStepSize(LearningRate):
  """Approximate Polyak Step Size

  The true Polyak step size, assuming f(x*) is known, is

    ( f(x) - f(x*) )/||\nabla f(x)||_2^2

  When f(x*) isn't known, the following still works,

    (f(x_t) - min_{i<=t}[f(x_i)] + gamma_t) / ||\nabla f(x_t)||_2^2

  When gamma_t = a/(b+iter)^p is non-summable but square-summable. The latter is
  implemented here. To imitate the former, specify f_min and set a = 0.0

  Parameters
  ----------
  a : float
      numerator of gamma_t
  b : float
      offset in denominator of gamma_t
  p : float
      power in the denominator of gamma_t
  f_min : float
      greater than or equal to f(x*)
  """

  def __init__(self, a=1.0, b=1.0, p=1.0, f_min=np.inf):
    self.a = a
    self.b = b
    self.p = p
    self.f_min = f_min

  def learning_rate(self, x, direction, objective, iteration, **kwargs):
    a, b, p = self.a, self.b, self.p
    f_min = min(objective(x), self.f_min)
    gamma = a / ((b + iteration) ** p)
    alpha = (objective(x) - f_min + gamma) / (np.linalg.norm(direction) ** 2)
    self.f_min = f_min
    return alpha
