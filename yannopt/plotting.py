import os

import numpy as np
import pylab as pl


def plot_iterates_vs_function(iterates, function, path=None, y_star=None):
  """Plot x_t vs. f(x_t)"""
  xmin, xmax = limits(iterates)
  evals = [function(x) for x in iterates]
  ymin, ymax = limits(evals + [y_star])

  pl.figure()
  pl.plot(np.linspace(xmin, xmax, 100), function(np.linspace(xmin, xmax, 100)), alpha=0.2)
  pl.scatter(iterates, evals, c=np.linspace(0.2, 0.8, len(iterates)))
  if y_star is not None:
    pl.plot([xmin, xmax], [y_star, y_star], 'r--', alpha=0.2)
  pl.xlabel('iterates')
  pl.ylabel('objective function evaluated at iterates')
  pl.xlim([xmin, xmax])
  pl.ylim([ymin, ymax])
  if path is not None:
    savefig(path)
    pl.close()


def plot_iteration_vs_function(iterates, function, path=None, y_star=None):
  """Plot t vs. f(x_t)"""
  xmin, xmax = -1, len(iterates)
  evals = [function(x) for x in iterates]
  ymin, ymax = limits(evals + [y_star])

  # plot iteration number vs. objective function evaluated at that iteration
  pl.figure()
  pl.plot(range(len(iterates)), evals, 'o')
  pl.plot(range(len(iterates)), evals, '--', alpha=0.2)
  if y_star is not None:
    pl.plot([xmin, xmax], [y_star, y_star], 'r--', alpha=0.2)
  pl.xlabel('iteration')
  pl.ylabel('value of objective function')
  pl.xlim([-1, len(iterates)])
  pl.ylim([ymin, ymax])
  if path is not None:
    savefig(path)
    pl.close()


def savefig(path):
  """Save a figure to a path and make a folder if it's not there"""
  try:
    import os
    folder, filename = os.path.split(path)
    os.makedirs(folder)
  except OSError:
    pass
  pl.savefig(path)


def limits(iterates):
  """find (xmin, xmax) spaced out by 10%"""
  iterates = filter(lambda x: not x is None, iterates)
  xmin, xmax = min(iterates), max(iterates)
  xmid = xmin + (xmax - xmin) / 2.0
  xmin, xmax = (np.array([xmin, xmax]) - xmid) * 1.2 + xmid
  return xmin, xmax
