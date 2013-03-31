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
      np.log(np.exp(X[0].dot(w))/(1 + np.exp(X[0].dot(w)))) + \
      np.log(                1.0/(1 + np.exp(X[1].dot(w))))
  )

  # gradient
  y_pred = np.array([
      1.0 / (1 + np.exp(-1 * X[0].dot(w))),
      1.0 / (1 + np.exp(-1 * X[1].dot(w))),
  ])
  assert_allclose(
      f.gradient(w),
      (y - y_pred)[0] * X[0] + (y - y_pred)[1] * X[1]
  )

  # hessian
  assert_allclose(
      f.hessian(w),
      y_pred[0] * (1-y_pred[0]) * np.outer(X[0], X[0])
        + y_pred[1] * (1-y_pred[1]) * np.outer(X[1], X[1])
  )
