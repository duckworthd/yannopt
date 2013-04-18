import numpy as np

from .basic import Quadratic
from ..base import Function
from .interfaces import Prox, Conjugate


class SquaredL2Norm(Quadratic):
  """(1/2)||Ax - b||_2^2"""

  def __init__(self, A=None, b=None, n=None):
    if n is None:
      n = A.shape[1]

    if A is None:
      A = np.eye(n)

    if b is None:
      b = np.zeros(n)

    # use (1/2)||Ax-b||_2^2
    #     = (1/2) (Ax-b)'(Ax-b)
    #     = (1/2) (x'A'Ax -2b'Ax + b'b)
    #     = (1/2) x'Qx - r'x + (1/2) s
    Q = A.T.dot(A)
    r = -b.dot(A)
    s = 0.5 * b.dot(b)

    Quadratic.__init__(self, A=Q, b=r, c=s)


class L1Norm(Prox, Conjugate, Function):
  """||x||_1"""

  def eval(self, x):
    return np.sum(np.abs(x))

  def gradient(self, x):
    return np.sign(x)

  def hessian(self, x):
    raise NotImplementedError("Hessian not defined")

  def prox(self, x, eta):
    # argmin_{y} eta ||y||_1 + 0.5 ||y-x||_2^2
    # -> 0 \in eta (d/dy)||y||_1 + y - x
    return np.maximum(x - eta, 0) - np.maximum(-x - eta, 0)

  @property
  def conjugate(self):
    # f^{*}(y) = max_{x} x'y - ||x||_1
    raise NotImplementedError("Hessian not defined")


class L2Norm(Prox, Conjugate, Function):

  def eval(self, x):
    raise NotImplementedError("Hessian not defined")

  def gradient(self, x):
    raise NotImplementedError("Hessian not defined")

  def hessian(self, x):
    raise NotImplementedError("Hessian not defined")

  def prox(self, x, eta):
    # argmin_{y} eta ||y||_2 + 0.5 ||y-x||_2^2
    raise NotImplementedError("Hessian not defined")

  @property
  def conjugate(self):
    # f^{*}(y) = max_{x} y'x - f(y)
    raise NotImplementedError("Hessian not defined")


class LInfinityNorm(Prox, Conjugate, Function):

  def eval(self, x):
    raise NotImplementedError("Hessian not defined")

  def gradient(self, x):
    raise NotImplementedError("Hessian not defined")

  def hessian(self, x):
    raise NotImplementedError("Hessian not defined")

  def prox(self, x, eta):
    # argmin_{y} eta ||y||_2 + 0.5 ||y-x||_2^2
    raise NotImplementedError("Hessian not defined")

  @property
  def conjugate(self):
    # f^{*}(y) = max_{x} y'x - f(y)
    raise NotImplementedError("Hessian not defined")


class Entropy(Prox, Conjugate, Function):
  def eval(self, x):
    raise NotImplementedError("Hessian not defined")

  def gradient(self, x):
    raise NotImplementedError("Hessian not defined")

  def hessian(self, x):
    raise NotImplementedError("Hessian not defined")

  def prox(self, x, eta):
    # argmin_{y} eta ||y||_2 + 0.5 ||y-x||_2^2
    raise NotImplementedError("Hessian not defined")

  @property
  def conjugate(self):
    # f^{*}(y) = max_{x} y'x - f(y)
    raise NotImplementedError("Hessian not defined")
