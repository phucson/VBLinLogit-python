"""High-dimensional linear regression example for vb_linear_*.

This script demonstrates the use of variational Bayesian linear regression
on a high-dimensional dataset with little training data. In this case, the
Bayesian shrinkage regularization should outperform maximum likelihood
estimates without shrinkage. The generated dataset has ~1.5 training
examples per dimension of the input.

Copyright (c) 2014-2019, Jan Drugowitsch (original MATLAB code)
All rights reserved.
See the file LICENSE for licensing information.
"""

import numpy as np

from _helpers import finish, ols, parse_args, style_axes
from vblinlogit import vb_linear_fit, vb_linear_pred


def main(save_dir=None, show=True, seed=0):
    import matplotlib.pyplot as plt

    # set RNG seed and plot limits
    rng = np.random.default_rng(seed)
    wlims = (-5, 5)
    ylims = (-11, 11)

    # settings
    D = 100             # dimensionality of the input
    N = 150             # size of the training set
    N_test = 50         # size of the test set

    # create data
    w = rng.standard_normal(D)
    X = rng.random((N, D)) - 0.5
    X_test = rng.random((N_test, D)) - 0.5
    y = X @ w + rng.standard_normal(N)
    y_test = X_test @ w + rng.standard_normal(N_test)

    # perform regression and make predictions
    # variational bayes linear regression, performance on train & test set
    w_VB, V_VB, _, _, an_VB, bn_VB = vb_linear_fit(X, y)[:6]
    y_VB = vb_linear_pred(X, w_VB, V_VB, an_VB, bn_VB)[0]
    y_test_VB, lam_VB, nu_VB = vb_linear_pred(X_test, w_VB, V_VB, an_VB, bn_VB)
    # maximum likelihood, performance on train & test set
    w_ML, wint_ML = ols(X, y)
    y_ML = X @ w_ML
    y_test_ML = X_test @ w_ML
    # output train and test set error
    mse = dict(train_ML=np.mean((y - y_ML) ** 2),
               train_VB=np.mean((y - y_VB) ** 2),
               test_ML=np.mean((y_test - y_test_ML) ** 2),
               test_VB=np.mean((y_test - y_test_VB) ** 2))
    print(f"Training set MSE: ML = {mse['train_ML']:f}, "
          f"VB = {mse['train_VB']:f}")
    print(f"Test     set MSE: ML = {mse['test_ML']:f}, "
          f"VB = {mse['test_VB']:f}")

    # plot coefficient estimates
    f1, ax = plt.subplots()
    ax.set_xlim(wlims)
    ax.set_ylim(wlims)
    # error bars
    for i in range(D):
        ax.plot(w[i] * np.ones(2) - 0.01,
                w_VB[i] + np.sqrt(V_VB[i, i]) * 1.96 * np.array([-1, 1]), '-',
                lw=0.25, color=(0.8, 0.5, 0.5))
        if wint_ML is not None:
            ax.plot(w[i] * np.ones(2) + 0.01, wint_ML[i, :], '-',
                    lw=0.25, color=(0.5, 0.5, 0.8))
    # means
    h1, = ax.plot(w - 0.01, w_VB, 'o', ms=3, mfc=(0.8, 0, 0), mec='none')
    h2, = ax.plot(w + 0.01, w_ML, '+', ms=3, mec=(0, 0, 0.8), mew=1)
    xymin = min(ax.get_xlim()[0], ax.get_ylim()[0])
    xymax = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([xymin, xymax], [xymin, xymax], 'k--', lw=0.5)
    ax.legend([h1, h2], ['VB', 'ML'])
    style_axes(ax)
    ax.set_xlabel('$w$')
    ax.set_ylabel('$w_{ML}$, $w_{VB}$')

    # plot test set predictions
    f2, ax = plt.subplots()
    ax.set_xlim(ylims)
    ax.set_ylim(ylims)
    y_VB_sd = np.sqrt(nu_VB / (lam_VB * (nu_VB - 2)))
    for i in range(N_test):
        ax.plot(y_test[i] * np.ones(2),
                y_test_VB[i] + y_VB_sd[i] * 1.96 * np.array([-1, 1]), '-',
                lw=0.25, color=(0.8, 0.5, 0.5))
    h1, = ax.plot(y_test, y_test_VB, 'o', ms=3, mfc=(0.8, 0, 0), mec='none')
    h2, = ax.plot(y_test, y_test_ML, '+', ms=3, mec=(0, 0, 0.8), mew=1)
    xymin = min(ax.get_xlim()[0], ax.get_ylim()[0])
    xymax = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([xymin, xymax], [xymin, xymax], 'k--', lw=0.5)
    ax.legend([h1, h2], ['VB', 'ML'])
    style_axes(ax)
    ax.set_xlabel('$y$')
    ax.set_ylabel('$y_{ML}$, $y_{VB}$')

    finish([f1, f2], 'vb_linear_example_highdim', save_dir, show)
    return mse


if __name__ == '__main__':
    main(**parse_args(__doc__.splitlines()[0]))
