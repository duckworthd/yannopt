"""
Common loss functions
"""
import itertools

import numpy as np

from .base import Function
from .utils import drop_dimensions


################################## Interfaces ##################################
class Prox(object):
  """A function that implements the prox operator

    prox_{eta}(x) = argmin_{y} eta f(y) + (1/2) ||y-x||_2^2
  """

  def prox(self, x, eta):
    raise NotImplementedError("Prox function not implemented")


################################## Classes #####################################
class LogisticLoss(Function):
  """Logistic Regression loss function

    \sum_{i} log(1 + exp(-y_i x_i' w))

  Parameters
  ----------
  X : [n_samples, n_features] array-like
      feature matrix
  y : [n_samples] array-like
      labels for each sample. Must be in {0, 1}
  """

  def __init__(self, X, y):
    self.X = np.atleast_2d(X)
    self.y = np.atleast_1d(y)

  def eval(self, x):
    X, y = self.X, self.y
    denominators = np.log(1 + np.exp(X.dot(x)))
    numerators = y * X.dot(x)
    return -1 * np.sum(numerators - denominators)

  def gradient(self, x):
    # gradient[f](w) = \sum_{i} (y_i - P(y=1|x;w)) x_i
    X, y = self.X, self.y
    y_hat = 1.0 / (1 + np.exp(-1 * X.dot(x)))
    return -1 * np.sum((y - y_hat)[:, np.newaxis] * X, axis=0)

  def hessian(self, x):
    # hessian[f](w) = \sum_{i} P(y=1|x;w) P(y=0|x;w) x_i x_i'
    n = len(x)
    X, y = self.X, self.y

    result = np.zeros((n, n))
    y_hat = 1.0 / (1 + np.exp(-1 * X.dot(x)))
    for (i, y_pred) in enumerate(y_hat):
      result += y_pred * (1.0 - y_pred) * np.outer(X[i], X[i])
    return result


class HingeLoss(Function):
  """SVM's Hinge loss function

    \sum_{i} max(0, 1 - y_i x_i' w)

  Parameters
  ----------
  X : [n_samples, n_features] array-like
      feature matrix
  y : [n_samples] array-like
      labels for each sample. Must be in {0, 1}
  """
  def __init__(self, X, y):
    self.X = np.atleast_2d(X)
    self.y = np.atleast_1d(y)

  def eval(self, x):
    X, y = self.X, self.y
    y = 2 * y - 1
    losses = np.maximum(0, 1 - y * X.dot(x))
    return np.sum(losses)

  def gradient(self, x):
    # gradient[f](w) = \sum_{i} 1[1 - y_i x_i'w > 0] -1 * y_i x_i
    X, y = self.X, self.y
    y = 2 * y - 1
    losses = np.maximum(0, 1 - y * X.dot(x))
    return np.sum(((losses > 0) * y)[:, np.newaxis] * X, axis=0)


class Quadratic(Prox, Function):
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
    A, b = self.A, self.b
    n = len(x)
    return np.linalg.lstsq(np.eye(n) + eta * A, x - eta * b)[0]


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
    return x


class Addition(Function):
  """Addition of functions

  Function of the form,

      f(x) = \sum_{i} f_{i}(x)
  """

  def __init__(self, functions):
    self.functions = functions

  def eval(self, x):
    evals = [f(x) for f in self.functions]
    return sum(evals)

  def gradient(self, x):
    gradients = [f.gradient(x) for f in self.functions]
    return sum(gradients)

  def hessian(self, x):
    hessians = [f.hessian(x) for f in self.functions]
    return sum(hessians)


class ElementwiseProduct(Function):
  """Elementwise Product of functions

  Function of the form,

      f(x) = \prod_{i} f_{i}(x)
  """

  def __init__(self, functions):
    self.functions = functions

  def eval(self, x):
    evals = [f(x) for f in self.functions]
    return np.prod(evals, axis=0)

  def gradient(self, x):
    # [df(x)/dx_k]_l = \sum_{i} (\prod_{j != i} [f_{j}(x)]_l) [df(x)/dx_k]_l
    gradients = [f.gradient(x) for f in self.functions]
    gradients = [np.atleast_2d(g).T for g in gradients]   # rows = outputs, columns = inputs

    # cross_products[i,j] = \prod_{k != i} [ f_{k}(x) ]_{j}
    evals = [f(x) for f in self.functions]
    prod  = np.prod(evals, axis=0)
    cross_products = [prod / e for e in evals]
    cross_products = np.nan_to_num(np.asarray(cross_products, dtype=float))

    result = 0
    for (cp, g) in zip(cross_products, gradients):
      result += np.atleast_2d(cp).T * g
    return drop_dimensions(result.T)

  def hessian(self, x):
    evals     = [f(x) for f in self.functions]
    gradients = [f.gradient(x) for f in self.functions]
    gradients = [np.atleast_2d(g).T for g in gradients]   # rows = outputs, columns = inputs
    hessians  = [f.hessian(x) for f in self.functions]

    n = len(x)
    n_func = len(self.functions)
    m = len(np.asarray(evals[0]))

    result = np.zeros( (m, n, n) )
    cross_products = self._cross_evals(x, 1)
    cross_products2= self._cross_evals(x, 2)
    for h in range(m):
      for i, j in itertools.product(range(n), repeat=2):
        for k in range(n_func):
          result[h,i,j] += hessians[k][h,i,j] * cross_products[k][h]
        for k, m in itertools.product(range(n_func), repeat=2):
          if k == m: continue
          result[h,i,j] += gradients[k][h,j] * gradients[m][h,i] * cross_products2[k,m][h]
    return result

  def _cross_evals(self, x, k):
    evals  = [np.asarray(f(x)) for f in self.functions]
    prod   = np.prod(evals, axis=0)
    n_func = len(self.functions)
    m      = len(np.asarray(evals[0]))
    result = np.zeros( [n_func]*k + [m] )
    for idx in itertools.product(range(n_func), repeat=k):
      result[idx] = prod / np.prod([evals[i] for i in idx], axis=0)
    return np.nan_to_num(result)



class Composition(Function):
  """f(g(x))

  Parameters
  ----------
  outer_function : Function
      components of f
  inner_function : Function
      components of g
  """

  def __init__(self, outer_function, inner_function):

    self.outer_function = outer_function
    self.inner_function = inner_function

  def eval(self, x):
    f,g = self.outer_function, self.inner_function
    y   = g(x)
    z   = f(y)
    return z

  def gradient(self, x):
    f,g = self.outer_function, self.inner_function
    y   = g(x)
    G1  = np.atleast_2d(f.gradient(y).T)
    H1  = np.atleast_2d(g.gradient(x).T)
    return drop_dimensions( ( G1.dot(H1) ).T )

  def hessian(self, x):
    f,g = self.outer_function, self.inner_function
    n = len(x)
    y = g(x)
    m = len(y)
    o = len(np.atleast_1d(self(x)))
    R = np.zeros( (o, n, n) )

    G2 = np.atleast_3d(g.hessian(x).T).T
    F2 = np.atleast_3d(f.hessian(y).T).T
    G1 = np.atleast_2d(g.gradient(x).T)
    F1 = np.atleast_2d(f.gradient(y).T)

    for a in range(o):
      for i, j in itertools.product(range(n), repeat=2):
        for k in range(m):
          R[a, i, j] += F1[a,k] * G2[k,i,j]
        for k, l in itertools.product(range(m), repeat=2):
          R[a, i, j] += F2[a,k,l] * G1[k,i] * G1[l,j]

    return drop_dimensions(R.T).T


class Stacked(Function):
  """[f_{1}(x); f_{2}(x); ...]"""

  def __init__(self, functions):
    self.functions = functions

  def eval(self, x):
    return np.hstack(f(x) for f in self.functions)

  def gradient(self, x):
    gradients = [f.gradient(x).T for f in self.functions]
    return drop_dimensions(np.vstack(gradients).T)

  def hessian(self, x):
    hessians = [f.hessian(x) for f in self.functions]
    hessians = [np.atleast_3d(h.T).T for h in hessians]
    result = np.concatenate(hessians, axis=0)
    return drop_dimensions(result.T).T


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


class L1Norm(Prox, Function):
  """||x||_1"""

  def eval(self, x):
    return np.sum(np.abs(x))

  def gradient(self, x):
    return np.sign(x)

  def hessian(self, x):
    raise NotImplementedError("Hessian not defined")

  def prox(self, x, eta):
    return np.maximum(x - eta, 0) - np.maximum(-x - eta, 0)


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
