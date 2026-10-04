"""Shared numerical helpers.

Copyright (c) 2013-2019, Jan Drugowitsch (original MATLAB code)
All rights reserved.
See the file LICENSE for licensing information.
"""

import numpy as np


class MaxIterWarning(UserWarning):
    """Issued when an iterative fit reaches its maximum number of iterations.

    Python counterpart of MATLAB's ``'Bayes:maxIter'`` warning identifier.
    """


def logdet(A):
    """Compute log(det(A)) of a positive definite matrix A.

    This is more accurate than ``np.log(np.linalg.det(A))``.
    """
    return 2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(A))))


def lam(xi):
    """Return 1 / (4 * xi) * tanh(xi / 2), with the limit 1/8 at xi = 0.

    Works for both scalars and arrays; returns the same kind as given.
    """
    xi_arr = np.asarray(xi, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.tanh(xi_arr / 2) / (4 * xi_arr)
    # fix values where xi = 0
    out = np.where(np.isnan(out), 1 / 8, out)
    if np.ndim(xi) == 0:
        return float(out)
    return out


def as_matrix(X):
    """Convert X to a 2D float array (N x D)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, np.newaxis]
    return X


def as_vector(y):
    """Convert y to a 1D float array."""
    return np.asarray(y, dtype=float).reshape(-1)
