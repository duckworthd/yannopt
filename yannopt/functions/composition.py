"""
`Function`s for combining other `Function` objects together
"""
import itertools

import numpy as np

from ..base import Function
from ..utils import drop_dimensions


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


