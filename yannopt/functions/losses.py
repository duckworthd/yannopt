import numpy as np

from ..base import Function


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
