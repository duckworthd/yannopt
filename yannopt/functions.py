"""
Common loss functions
"""
import numpy as np

from .base import Function


################################## Interfaces ##################################
class Prox(Function):
  """A function that implements the prox operator

    prox_{eta}(x) = min_{y} f(y) + (1/2 eta) ||y-x||_2^2
  """

  def prox(x, eta):
    raise NotImplementedError("Prox function not implemented")


################################## Classes #####################################
class LogisticRegression(Function):
  """Logistic Regression loss function

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
  """Quadratic function

  min_{x} 0.5 x'Ax + b'x + c

  Parameters
  ----------
  A : [n, n] array-like
  b : [n] array-like
  c : [1] array-like
  """

  def __init__(self, A, b, c=0.0):
    self.A = np.asarray(A)
    self.b = np.asarray(b)
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


class Separable(Function):
  """A separable function

  Function of the form,

      f(x) = \sum_{i} f_{i}(x)
  """

  def __init__(self, functions):
    self.functions = functions

  def eval(self, x):
    evals = [f(x) for f in self.functions]
    return np.sum(evals)

  def gradient(self, x):
    gradients = [f.gradient(x) for f in self.functions]
    return np.sum(gradients)

  def hessian(self, x):
    hessians = [f.gradient(x) for f in self.functions]
    return np.sum(hessians)


################################## Functions ###################################
def quadratic_approx(f, x):
  """Compute quadratic approximation to a smooth function

    f(y) ~~ f(x) + g'(y-x) + (1/2 lmbda)(y-x)'H(y-x)

  where g = gradient[f](x)
        H =  hessian[f](x)
  """
  c = f(x)
  g = f.gradient(x)
  H = f.hessian(x)

  return Quadratic(H, g, c)
