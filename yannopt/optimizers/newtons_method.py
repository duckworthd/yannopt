"""
Newton's Method
"""

import numpy as np

from ..base import Optimizer
from ..constraints.base import is_linear


class NewtonsMethod(Optimizer):
  """Vanilla Implementation of Newton's 2nd order optimizer for unconstrained problems"""

  def optimize(self, objective, x0):
    kwargs = {
        'objective': objective,
    }
    iteration = 0
    x = x0

    while True:
      if self.stopping_criterion(iteration=iteration, x=x, **kwargs):
        break
      else:
        # Solve the following approximation to the original problem
        #
        # min_{y} f(x) + g'y + (1/2) y'Hy
        #
        # where g = gradient[f](x)
        #       H =  hessian[f](x)
        H = objective.hessian(x)
        g = objective.gradient(x)
        direction = -1 * np.linalg.lstsq(H, g)[0]
        eta = self.learning_rate(iteration=iteration, x=x,
            direction=direction, **kwargs)

        x  += eta * direction

        iteration += 1

    return x


class QPNewtonsMethod(Optimizer):
  """Solver for Quadratic Programs with equality constraints

  min_{x} (1/2) x'Qx + c'x
  s.t.    Ax = b
  """

  def optimize(self, objective, x0=None):
    if not is_linear(objective.equality_constraint):
      ### UNCONSTRAINED ###
      # use the KKT condition on the gradient of the Lagrangian = 0
      # (implies Qx + c = 0
      Q, c = objective.objective.A, objective.objective.b
      x = np.linalg.solve(Q, -c)
      return x
    else:
      ### CONSTRAINED ###
      # Use the same KKT condition, but on a bigger problem.
      #
      # [ Q A' ] [ x*] = [-c ]
      # [ A 0  ] [ v ] = [ b ]
      #
      # where v is the Lagrange multiplier for Ax = b
      assert (len(objective.inequality_constraints) == 1,
          "Only one equality constraint allowed")
      Q, c = objective.objective.A, objective.objective.b
      A, b = objective.equality_constraint.A, objective.equality_constraint.b

      n_v, n_x = A.shape

      M = np.zeros((n_x + n_v, n_x + n_v))
      M[0:n_x, 0:n_x]           = Q
      M[n_x:n_x+n_v, 0:n_x]     = A
      M[0:n_x, n_x:n_x+n_v]     = A.T

      r = np.hstack([-c, b])

      y = np.linalg.lstsq(M, r)[0]
      x = y[0:n_x]
      # v = y[n_x:n_x+n_v]
      return x


class LinearConstrainedNewtonsMethod(Optimizer):
  """Newton's Method with equality constraints

  min_{x} f_{0}(x)
  s.t.    Ax = b
  """

  def optimize(self, objective, x0):
    assert len(objective.inequality_constraints) == 0, "This method cannot handle inequality constraints"
    A, b = objective.equality_constraint.A, objective.equality_constraint.b
    objective = objective.objective
    kwargs = {
        'objective': objective,
    }
    iteration = 0
    x = x0

    while True:
      if self.stopping_criterion(iteration=iteration, x=x, **kwargs):
        break
      else:
        # solve the KKT conditions matrix for the following quadratic
        # approximation to the original problem
        #
        # min_{y} f(x) + g'y + (1/2)y'Hy
        # s.t.    A(x + y) = b
        #
        # where g = gradient[f](x)
        #       H =  hessian[f](x)
        #
        # [ H A' ] [ y ] = [-g ]
        # [ A 0  ] [ v ] = [ 0 ]
        #
        # y is then a search direction 
        H = objective.hessian(x)
        g = objective.gradient(x)

        n_v, n_x = A.shape

        M = np.zeros((n_x + n_v, n_x + n_v))
        M[0:n_x, 0:n_x]           = H
        M[n_x:n_x+n_v, 0:n_x]     = A
        M[0:n_x, n_x:n_x+n_v]     = A.T

        r = np.hstack([-g, np.zeros(b.shape)])
        y = np.linalg.lstsq(M, r)[0]
        direction = y[0:n_x]

        eta = self.learning_rate(iteration=iteration, x=x,
            direction=direction, **kwargs)

        x  += eta * direction

        iteration += 1

    return x
