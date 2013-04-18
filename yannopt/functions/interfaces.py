"""
Mixin annotations for `Function`s
"""

class Prox(object):
  """A function that implements the prox operator

    prox_{eta}(x) = argmin_{y} eta f(y) + (1/2) ||y-x||_2^2
  """

  def prox(self, x, eta):
    raise NotImplementedError("Prox function not implemented")


class Conjugate(object):
  """A function whose Fenchel conjugate can be computed easily

    f^{*}(y) = \sup_{x} y^T x - f(y)
  """

  @property
  def conjugate(self):
    raise NotImplementedError("Conjugate function not implemented")


