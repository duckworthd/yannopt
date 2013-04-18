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


