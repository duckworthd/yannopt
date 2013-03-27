"""
Common loss functions
"""
import numpy as np

from .base import Function


class LogisticRegression(Function):
  """Logistic Regression problem

  min_{w} \sum_{i} log(1 + exp(-y_i x_i' w))

  Parameters
  ----------
  X : [n_samples, n_features] array-like
      feature matrix
  y : [n_samples] array-like
      labels for each sample. Must be in {-1, 1}
  """

  def __init__(self, X, y):
    self.X = np.atleast_2d(X)
    self.y = np.atleast_1d(y)

  def eval(self, x):
    X, y = self.X, self.y
    loss = np.log(1 + np.exp(-y * X.dot(x)))
    return np.sum(loss)

  def gradient(self, x):
    X, y = self.X, self.y
    y_hat = 1.0 / (1 + np.exp(-1 * X.dot(x)))
    return np.sum(y - y_hat) * x

  def hessian(self, x):
    raise NotImplementedError("TODO")


class Quadratic(Function):
  """An unconstrained Quadratic Program

  min_{x} 0.5 x'Ax + b'x

  Parameters
  ----------
  A : [n, n] array-like
  b : [n] array-like
  """

  def __init__(self, A, b):
    self.A = np.asarray(A)
    self.b = np.asarray(b)

  def eval(self, x):
    A, b = self.A, self.b
    return 0.5 * x.dot(A).dot(x) + b.dot(x)

  def gradient(self, x):
    A, b = self.A, self.b
    return A.dot(x) + b

  def hessian(self, x):
    return self.A

  def solution(self):
    A, b = self.A, self.b
    return np.linalg.solve(A, -b)


class Constant(Function):

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
