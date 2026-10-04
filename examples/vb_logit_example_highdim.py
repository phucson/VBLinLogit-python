"""High-dimensional logistic regression, using vb_logit_*.

This example demonstrates the ability of automated relevance determination
(ARD) to detect and ignore irrelevant input dimensions for logistic
regression. The script shows this by generating an input -> output mapping
with a D-dimensional input space, of which only a small subset of D_eff
dimensions determine the output class. It then compares variational
Bayesian linear regression without and with ARD, and linear discriminant
analysis, and shows that the variant with ARD is better able to estimate the
regression coefficients, and also provides lower-error predictions on the
test set.

Copyright (c) 2014-2019, Jan Drugowitsch (original MATLAB code)
All rights reserved.
See the file LICENSE for licensing information.
"""

import numpy as np

from _helpers import finish, parse_args, style_axes
from vblinlogit import (vb_logit_fit, vb_logit_fit_ard, vb_logit_fit_iter,
                        vb_logit_pred)


def main(save_dir=None, show=True, seed=0):
    import matplotlib.pyplot as plt

    # set RNG seed and plot limits
    rng = np.random.default_rng(seed)
    wlims = (-4, 4)

    # settings
    D = 1000         # full input dimensionality
    D_eff = 100      # number of effective input dimensions
    N = 2000         # number of training set examples
    N_test = 10000   # number of test set examples
    N_plot = 200     # number of examples to plot

    # generate data
    w = np.concatenate([rng.standard_normal(D_eff), np.zeros(D - D_eff)])
    # inputs for train & test set
    X = rng.random((N, D)) - 0.5
    X_test = rng.random((N_test, D)) - 0.5
    # output probabilities and samples for train & test set
    py = 1 / (1 + np.exp(-X @ w))
    y = 2.0 * (rng.random(N) < py) - 1
    py_test = 1 / (1 + np.exp(-X_test @ w))
    y_test = 2.0 * (rng.random(N_test) < py_test) - 1

    # estimate coefficients, form predictions for train & test set
    # VB, no ARD
    print('Variational Bayesian estimation, no ARD')
    w_VB, V_VB, invV_VB = vb_logit_fit(X, y)[:3]
    py_VB = vb_logit_pred(X, w_VB, V_VB, invV_VB)
    py_test_VB = vb_logit_pred(X_test, w_VB, V_VB, invV_VB)
    # VB, no hyper-priors, no ARD
    print('Variational Bayesian estimation, no ARD, no hyper-priors')
    w_VB1, V_VB1, invV_VB1 = vb_logit_fit_iter(X, y)[:3]
    py_VB1 = vb_logit_pred(X, w_VB1, V_VB1, invV_VB1)
    py_test_VB1 = vb_logit_pred(X_test, w_VB1, V_VB1, invV_VB1)
    # VB, ARD
    print('Variational Bayesian estimation, with ARD')
    w_VB2, V_VB2, invV_VB2 = vb_logit_fit_ard(X, y)[:3]
    py_VB2 = vb_logit_pred(X, w_VB2, V_VB2, invV_VB2)
    py_test_VB2 = vb_logit_pred(X_test, w_VB2, V_VB2, invV_VB2)
    # Fisher linear discriminant analysis
    print('Fisher linear discriminant analysis')
    y1 = y == 1
    m1, m0 = X[y1].mean(axis=0), X[~y1].mean(axis=0)
    w_LD = np.linalg.solve(np.cov(X[y1], rowvar=False)
                           + np.cov(X[~y1], rowvar=False), m1 - m0)
    c_LD = 0.5 * (m1 + m0) @ w_LD
    y_LD = 2 * (X @ w_LD > c_LD) - 1
    y_test_LD = 2 * (X_test @ w_LD > c_LD) - 1
    # output train and test-set error
    loss = dict(
        train_LDA=np.mean(y_LD != y),
        train_VB=np.mean(2 * (py_VB > 0.5) - 1 != y),
        train_VBiter=np.mean(2 * (py_VB1 > 0.5) - 1 != y),
        train_VB_ARD=np.mean(2 * (py_VB2 > 0.5) - 1 != y),
        test_LDA=np.mean(y_test_LD != y_test),
        test_VB=np.mean(2 * (py_test_VB > 0.5) - 1 != y_test),
        test_VBiter=np.mean(2 * (py_test_VB1 > 0.5) - 1 != y_test),
        test_VB_ARD=np.mean(2 * (py_test_VB2 > 0.5) - 1 != y_test))
    for s, pad in [('train', 'training'), ('test', 'test    ')]:
        print(f"{pad} set 0-1 loss: LDA    = {loss[s + '_LDA']:f}, "
              f"VB        = {loss[s + '_VB']:f}\n"
              f"                       VBiter = {loss[s + '_VBiter']:f}, "
              f"VB w/ ARD = {loss[s + '_VB_ARD']:f}")

    # plot coefficient estimates
    f1, ax = plt.subplots()
    ax.set_xlim(wlims)
    ax.set_ylim(wlims)
    # error bars
    pm = 1.96 * np.array([-1, 1])
    for i in range(D):
        ax.plot(w[i] * np.ones(2) - 0.03, w_VB[i] + np.sqrt(V_VB[i, i]) * pm,
                '-', lw=0.25, color=(0.8, 0.5, 0.5))
        ax.plot(w[i] * np.ones(2) - 0.01, w_VB1[i] + np.sqrt(V_VB1[i, i]) * pm,
                '-', lw=0.25, color=(0.8, 0.5, 0.8))
        ax.plot(w[i] * np.ones(2) + 0.01, w_VB2[i] + np.sqrt(V_VB2[i, i]) * pm,
                '-', lw=0.25, color=(0.5, 0.8, 0.5))
    # means
    h1, = ax.plot(w - 0.03, w_VB, 'o', ms=3, mfc=(0.8, 0, 0), mec='none')
    h2, = ax.plot(w - 0.01, w_VB1, 'd', ms=3, mfc=(0.8, 0, 0.8), mec='none')
    h3, = ax.plot(w + 0.01, w_VB2, 's', ms=3, mfc=(0, 0.8, 0), mec='none')
    h4, = ax.plot(w + 0.03, w_LD, '+', ms=3, mec=(0, 0, 0.8), mew=1)
    xymin = min(ax.get_xlim()[0], ax.get_ylim()[0])
    xymax = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([xymin, xymax], [xymin, xymax], 'k--', lw=0.5)
    ax.legend([h1, h2, h3, h4],
              ['VB', 'VB w/o hyperpriors', 'VB w/ ARD', 'LDA'])
    style_axes(ax)
    ax.set_xlabel('$w$')
    ax.set_ylabel('$w_{ML}$, $w_{VB}$')

    # plot test set predictions
    f2, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # misclassification areas
    ax.fill([0.5, 1, 1, 0.5], [0, 0, 0.5, 0.5], color=(0.95, 0.8, 0.8), lw=0)
    ax.fill([0, 0.5, 0.5, 0], [0.5, 0.5, 1, 1], color=(0.95, 0.8, 0.8), lw=0)
    # p(y=1) for N_plot samples of test set
    h1, = ax.plot(py_test[:N_plot], py_test_VB[:N_plot], 'o', ms=3,
                  mfc=(0.8, 0, 0), mec='none')
    h2, = ax.plot(py_test[:N_plot], py_test_VB1[:N_plot], 'd', ms=3,
                  mfc=(0.8, 0, 0.8), mec='none')
    h3, = ax.plot(py_test[:N_plot], py_test_VB2[:N_plot], 's', ms=3,
                  mfc=(0, 0.8, 0), mec='none')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.5)
    ax.legend([h1, h2, h3], ['VB', 'VB w/o hyperpriors', 'VB w/ ARD'])
    style_axes(ax)
    ax.set_xlabel('$p_{true}(y = 1)$')
    ax.set_ylabel('$p_{VB}(y = 1)$')

    finish([f1, f2], 'vb_logit_example_highdim', save_dir, show)
    return loss


if __name__ == '__main__':
    main(**parse_args(__doc__.splitlines()[0]))
