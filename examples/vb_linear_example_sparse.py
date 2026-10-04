"""Sparse linear regression example for vb_linear_*.

This example demonstrates the ability of automated relevance determination
(ARD) to detect and ignore irrelevant input dimensions. The script shows
this by generating an input -> output mapping with a D-dimensional input
space, of which only a small subset of D_eff dimensions determine the
output. It then compares variational Bayesian linear regression without and
with ARD, and a least-square estimate, and shows that the variant with ARD
is better able to estimate the regression coefficients, and also provides
lower-error predictions on the test set.

Copyright (c) 2014-2019, Jan Drugowitsch (original MATLAB code)
All rights reserved.
See the file LICENSE for licensing information.
"""

import numpy as np

from _helpers import finish, ols, parse_args, style_axes
from vblinlogit import vb_linear_fit, vb_linear_fit_ard, vb_linear_pred


def main(save_dir=None, show=True, seed=0):
    import matplotlib.pyplot as plt

    # set RNG seed and plot limits
    rng = np.random.default_rng(seed)
    wlims = (-5, 5)
    ylims = (-15, 15)

    # settings
    D = 1000         # full input dimensionality
    D_eff = 100      # number of effective input dimensions
    N = 500          # number of training set examples
    N_test = 50      # number of test set examples

    # create data
    w = np.concatenate([rng.standard_normal(D_eff), np.zeros(D - D_eff)])
    X = rng.random((N, D)) - 0.5
    X_test = rng.random((N_test, D)) - 0.5
    y = X @ w + rng.standard_normal(N)
    y_test = X_test @ w + rng.standard_normal(N_test)

    # perform regression and make predictions
    # variational bayes linear regression, train & test-set predictions
    w_VB, V_VB, _, _, an_VB, bn_VB = vb_linear_fit(X, y)[:6]
    y_VB = vb_linear_pred(X, w_VB, V_VB, an_VB, bn_VB)[0]
    y_test_VB, lam_VB, nu_VB = vb_linear_pred(X_test, w_VB, V_VB, an_VB, bn_VB)
    # variational bayes linear regression with ARD, train & test-set
    # predictions
    w_VB2, V_VB2, _, _, an_VB2, bn_VB2 = vb_linear_fit_ard(X, y)[:6]
    y_VB2 = vb_linear_pred(X, w_VB2, V_VB2, an_VB2, bn_VB2)[0]
    y_test_VB2, lam_VB2, nu_VB2 = vb_linear_pred(X_test, w_VB2, V_VB2,
                                                 an_VB2, bn_VB2)
    # maximum likelihood, train & test-set predictions (N < D, so this is
    # the minimum-norm least-squares solution, without confidence intervals)
    w_ML, wint_ML = ols(X, y)
    y_ML = X @ w_ML
    y_test_ML = X_test @ w_ML
    # output train and test set error
    mse = dict(train_ML=np.mean((y - y_ML) ** 2),
               train_VB=np.mean((y - y_VB) ** 2),
               train_VB_ARD=np.mean((y - y_VB2) ** 2),
               test_ML=np.mean((y_test - y_test_ML) ** 2),
               test_VB=np.mean((y_test - y_test_VB) ** 2),
               test_VB_ARD=np.mean((y_test - y_test_VB2) ** 2))
    print(f"Training set MSE: ML = {mse['train_ML']:f}, "
          f"VB = {mse['train_VB']:f}, VB w/ ARD = {mse['train_VB_ARD']:f}")
    print(f"Test     set MSE: ML = {mse['test_ML']:f}, "
          f"VB = {mse['test_VB']:f}, VB w/ ARD = {mse['test_VB_ARD']:f}")

    # plot coefficient estimates
    f1, ax = plt.subplots()
    ax.set_xlim(wlims)
    ax.set_ylim(wlims)
    # error bars
    pm = 1.96 * np.array([-1, 1])
    for i in range(D):
        ax.plot(w[i] * np.ones(2) - 0.02, w_VB[i] + np.sqrt(V_VB[i, i]) * pm,
                '-', lw=0.25, color=(0.8, 0.5, 0.5))
        ax.plot(w[i] * np.ones(2) + 0.02, w_VB2[i] + np.sqrt(V_VB2[i, i]) * pm,
                '-', lw=0.25, color=(0.5, 0.8, 0.5))
        if wint_ML is not None:
            ax.plot(w[i] * np.ones(2), wint_ML[i, :], '-', lw=0.25,
                    color=(0.5, 0.5, 0.8))
    # means
    h1, = ax.plot(w - 0.02, w_VB, 'o', ms=3, mfc=(0.8, 0, 0), mec='none')
    h2, = ax.plot(w + 0.02, w_VB2, 's', ms=3, mfc=(0, 0.8, 0), mec='none')
    h3, = ax.plot(w, w_ML, '+', ms=3, mec=(0, 0, 0.8), mew=1)
    xymin = min(ax.get_xlim()[0], ax.get_ylim()[0])
    xymax = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([xymin, xymax], [xymin, xymax], 'k--', lw=0.5)
    ax.legend([h1, h2, h3], ['VB', 'VB w/ ARD', 'ML'])
    style_axes(ax)
    ax.set_xlabel('$w$')
    ax.set_ylabel('$w_{ML}$, $w_{VB}$')

    # plot test set predictions
    f2, ax = plt.subplots()
    ax.set_xlim(ylims)
    ax.set_ylim(ylims)
    y_VB_sd = np.sqrt((nu_VB / (nu_VB - 2)) / lam_VB)
    y_VB2_sd = np.sqrt((nu_VB2 / (nu_VB2 - 2)) / lam_VB2)
    # error bars
    for i in range(N_test):
        ax.plot(y_test[i] * np.ones(2) - 0.02, y_test_VB[i] + y_VB_sd[i] * pm,
                '-', lw=0.25, color=(0.8, 0.5, 0.5))
        ax.plot(y_test[i] * np.ones(2) + 0.02,
                y_test_VB2[i] + y_VB2_sd[i] * pm,
                '-', lw=0.25, color=(0.5, 0.8, 0.5))
    # means
    h1, = ax.plot(y_test - 0.02, y_test_VB, 'o', ms=3, mfc=(0.8, 0, 0),
                  mec='none')
    h2, = ax.plot(y_test + 0.02, y_test_VB2, 's', ms=3, mfc=(0, 0.8, 0),
                  mec='none')
    h3, = ax.plot(y_test, y_test_ML, '+', ms=3, mec=(0, 0, 0.8), mew=1)
    xymin = min(ax.get_xlim()[0], ax.get_ylim()[0])
    xymax = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([xymin, xymax], [xymin, xymax], 'k--', lw=0.5)
    ax.legend([h1, h2, h3], ['VB', 'VB w/ ARD', 'ML'])
    style_axes(ax)
    ax.set_xlabel('$y$')
    ax.set_ylabel('$y_{ML}$, $y_{VB}$')

    finish([f1, f2], 'vb_linear_example_sparse', save_dir, show)
    return mse


if __name__ == '__main__':
    main(**parse_args(__doc__.splitlines()[0]))
