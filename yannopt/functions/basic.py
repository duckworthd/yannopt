"""
Commonly encountered functions
"""
import numpy as np

from ..base import Function
from .interfaces import Prox, Conjugate


class Quadratic(Prox, Conjugate, Function):
  """Quadratic function

  0.5 x'Ax + b'x + c

  Parameters
  ----------
  A : [n, n] array-like
      Symmetric, positive semidefinite matrix
  b : [n] array-like
  c : [1] array-like
  """

  def __init__(self, A, b, c=0.0):
    self.A = np.atleast_2d(A)
    self.b = np.atleast_1d(b)
    self.c = np.asarray(c)

  def eval(self, x):
    A, b, c = self.A, self.b, self.c
    return 0.5 * x.dot(A).dot(x) + b.dot(x) + c

  def gradient(self, x):
    A, b = self.A, self.b
    return A.dot(x) + b

  def hessian(self, x):
    return self.A

  def solution(self):
    A, b = self.A, self.b
    return np.linalg.solve(A, -b)

  def prox(self, x, eta):
    # argmin_{y} 0.5 y'Ay + b'y + c + 0.5/eta ||y-x||_2^2
    # -> Ay + b + (1/eta)(y-x) = 0
    # -> (A + (1/eta)I) y = (1/eta)x - b
    # -> (eta A + I) y = x - eta b
    A, b = self.A, self.b
    n = len(x)
    return np.linalg.lstsq(np.eye(n) + eta * A, x - eta * b)[0]

  @property
  def conjugate(self):
    A, b, c = self.A, self.b, self.c
    A2 = np.linalg.pinv(A)
    b2 = -A2.dot(b)
    c2 = 0.5 * b.dot(A2).dot(b) - c
    return Quadratic(A2, b2, c2)


class Affine(Function):
  """Affine function

    Ax + b

  Parameters
  ----------
  A : [m, n] array-like
  b : [m] array-like
  """

  def __init__(self, A, b):
    self.A = A
    self.b = b

  def eval(self, x):
    A, b = self.A, self.b
    return A.dot(x) + b

  def gradient(self, x):
    A, b = self.A, self.b
    return self.A.T

  def hessian(self, x):
    m,n = self.A.shape
    return np.zeros((m,n,n))


class Constant(Prox, Function):

  def __init__(self, c):
    self.c = c

  def eval(self, x):
    return self.c

  def gradient(self, x):
    n = x.shape[0]
    return np.zeros(n)

  def hessian(self, x):
    n = x.shape[0]
    return np.zeros((n,n))

  def prox(self, x, eta):
    # argmin_{y} c + 0.5/eta ||y - x||_2^2
    return x
