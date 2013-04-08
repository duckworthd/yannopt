"""
Miscellaneous functions
"""

import numpy as np
from numpy import linalg


def null_space(A, eps=1e-15):
  """Find V such that AVx = 0 for all x"""
  _, s, Vh = linalg.svd(A)
  s2 = np.zeros(Vh.shape[0])
  s2[0:len(s)] = s
  mask = s2 < eps
  null = Vh[mask,:]
  return null.T


def vectorize(args):
  """convert a dictionary of arrays to a single vector"""
  # determine length of vector
  size = np.sum(np.prod(a.shape) for a in args.values())
  x = np.zeros(size)
  sizes = {}
  i = 0
  for (k, v) in sorted(args.items()):
    s = np.prod(v.shape)
    x[i:i+s] = v.ravel()
    sizes[k] = v.shape
    i += s
  return sizes, x


def unvectorize(sizes, x):
  """convert a single vector into a dictionary of arrays"""
  result = {}
  i = 0
  for (key, shape) in sorted(sizes.items()):
    s = np.prod(shape)
    A = x[i:i+s].reshape(shape)
    result[key] = A
    i += s
  return result


def drop_dimensions(x):
  """If any final dimensions are 1, drop them

  For example, if x.shape == (3,5,1), then this will return an array of size
  (3,5)
  """
  while x.shape[-1] == 1:
    x = x.reshape(x.shape[0:-1])
  return x
