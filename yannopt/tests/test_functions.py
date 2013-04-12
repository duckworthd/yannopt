import itertools

import numpy as np
from numpy.testing import assert_allclose

from yannopt import functions


def test_logistic_loss():
  X = np.array([[ 1.0],
                [-1.0]])
  y = np.array([1, 0])

  f = functions.LogisticLoss(X, y)
  w = np.array([2.0])

  # eval
  assert_allclose(
      f(w),
      -np.log(np.exp(X[0].dot(w))/(1 + np.exp(X[0].dot(w)))) + \
      -np.log(                1.0/(1 + np.exp(X[1].dot(w))))
  ), 'logistic loss'

  # gradient
  y_pred = np.array([
      1.0 / (1 + np.exp(-1 * X[0].dot(w))),
      1.0 / (1 + np.exp(-1 * X[1].dot(w))),
  ])
  assert_allclose(
      f.gradient(w),
      -(y - y_pred)[0] * X[0] + \
      -(y - y_pred)[1] * X[1]
  ), 'logistic loss gradient'

  # hessian
  assert_allclose(
      f.hessian(w),
      y_pred[0] * (1-y_pred[0]) * np.outer(X[0], X[0])
        + y_pred[1] * (1-y_pred[1]) * np.outer(X[1], X[1])
  ), 'logistic loss hessian'


def test_hinge_loss():
  X = np.array([[ 1.0, 1.0],
                [-1.0, 2.0]])
  y = np.array([1, 1])

  f = functions.HingeLoss(X, y)
  w = np.array([2.0, -0.5])

  # eval
  assert_allclose(
      f(w),
      max(0, 1 - y[0] * X[0].dot(w)) + \
      max(0, 1 - y[1] * X[1].dot(w))
  ), 'hinge loss'

  # gradient
  assert_allclose(
    f.gradient(w),
    0 + \
    - 1 * -1 * X[1]
  ), 'hinge loss subgradient'


def test_stacked():
  A = np.array([[1,2],
                [4,5]])
  b = np.array([1])
  w = np.asarray([2,1])

  f1= functions.Affine(A, -b)
  f2= functions.Quadratic(A.T.dot(A), np.asarray([0,2]))
  f = functions.Stacked([f1, f2])

  # eval
  assert_allclose(
      f(w),
      list(f1(w)) + [f2(w)]
  )
  assert f(w).shape == (3,)

  # gradient
  assert_allclose(
      f.gradient(w),
      np.vstack([f1.gradient(w).T, f2.gradient(w)]).T
  )
  assert f.gradient(w).shape == (2, 3)

  # hessian
  assert_allclose(
      f.hessian(w),
      np.concatenate([f1.hessian(w), np.atleast_3d(f2.hessian(w)).T])
  )
  assert f.hessian(w).shape == (3, 2, 2)


def test_composition():
  A = np.array([[1,2,3],
                [4,5,6]])
  b = np.array([1,-1])
  x = np.array([5,9,19])

  g = functions.Affine(A, -b)
  f = functions.SquaredL2Norm(n=2)

  composed = functions.Composition(f, g)

  # eval
  assert_allclose(
      composed(x),
      0.5 * np.linalg.norm(A.dot(x) - b) ** 2
  )

  # gradient
  assert_allclose(
      composed.gradient(x),
      A.T.dot( A.dot(x) - b )
  )

  # hessian
  assert_allclose(
      composed.hessian(x),
      A.T.dot(A)
  )


def test_elementwise_multiplication():
  A = np.array([[1,2,3],
                [4,5,6]])
  b = np.array([1,-1])
  x = np.array([5,9,19])

  g = functions.Affine(A, -b)
  product = functions.ElementwiseProduct([g, g])

  # eval
  assert_allclose(
      product(x),
      g(x) * g(x)
  )

  # gradient
  assert_allclose(
      product.gradient(x),
      (2 * A * np.atleast_2d(g(x)).T).T
  )

  # hessian
  hessian = np.zeros( (2, 3, 3) )
  for i in range(2):
    for j, k in itertools.product(range(3), repeat=2):
      hessian[i,j,k] = 2 * A[i,j] * A[i,k]
  assert_allclose(
      product.hessian(x),
      hessian
  )
