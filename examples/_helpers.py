"""Helpers shared by the example scripts.

Copyright (c) 2014-2019, Jan Drugowitsch (original MATLAB code)
All rights reserved.
See the file LICENSE for licensing information.
"""

import os
import sys

import numpy as np
from scipy import stats

# make the vblinlogit package importable without installing it
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def ols(X, y, alpha=0.05):
    """Least-squares / maximum likelihood regression.

    Python counterpart of MATLAB's ``[b, bint] = regress(y, X)`` when there
    are more observations than regressors, returning the coefficients and
    their (1 - alpha) confidence intervals as a (D, 2) array. If X does not
    have full column rank, it returns the minimum-norm least-squares
    solution and ``None`` for the confidence intervals.
    """
    N, D = X.shape
    w, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    if rank < D or N <= D:
        return w, None
    dof = N - D
    s2 = np.sum((y - X @ w) ** 2) / dof
    se = np.sqrt(s2 * np.diag(np.linalg.inv(X.T @ X)))
    t = stats.t.ppf(1 - alpha / 2, dof)
    return w, np.column_stack([w - t * se, w + t * se])


def poly_basis(x, d):
    """(d-1)'th order polynomial basis, one row per element of x."""
    return np.asarray(x, dtype=float)[:, None] ** np.arange(d)


def fisher_lda_with_bias(X, y1):
    """Fisher LDA for inputs whose first column is a constant 1.

    Returns w such that sign(X @ w) gives the predicted class; w[0] is the
    bias term. y1 is a boolean vector marking the y = 1 class.
    """
    Z1, Z0 = X[y1, 1:], X[~y1, 1:]
    w = np.full(X.shape[1], np.nan)
    w[1:] = np.linalg.solve(np.cov(Z1, rowvar=False)
                            + np.cov(Z0, rowvar=False),
                            Z1.mean(axis=0) - Z0.mean(axis=0))
    w[0] = -0.5 * (Z1.mean(axis=0) + Z0.mean(axis=0)) @ w[1:]
    return w


def style_axes(ax, aspect=1.0, ticklen=0.02):
    """Mimic the MATLAB axis styling used in the original examples."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', length=ticklen * 250 / aspect)
    ax.set_box_aspect(1 / aspect)


def finish(figs, name, save_dir=None, show=True):
    """Save figures to save_dir (if given) and show them (if requested)."""
    import matplotlib.pyplot as plt
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        for i, fig in enumerate(figs, start=1):
            path = os.path.join(save_dir, f'{name}_{i}.png')
            fig.savefig(path, dpi=120, bbox_inches='tight')
            print(f'Saved {path}')
    if show:
        plt.show()
    else:
        for fig in figs:
            plt.close(fig)


def parse_args(description):
    """Command line options shared by all examples."""
    import argparse
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--no-show', action='store_true',
                        help='do not open figure windows')
    parser.add_argument('--save-dir', default=None,
                        help='directory to save figures to (as PNG)')
    args = parser.parse_args()
    if args.no_show:
        import matplotlib
        matplotlib.use('Agg')
    return dict(save_dir=args.save_dir, show=not args.no_show)
