"""
Miscellaneous functions
"""

from numpy import linalg


def null_space(A, eps=1e-15):
  """Find V such that AVx = 0 for all x"""
  _, s, Vh = linalg.svd(A)
  mask = s < eps
  null = Vh[mask,:]
  return null.T
