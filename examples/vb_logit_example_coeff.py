"""Estimating coefficients and separating hyperplane, using vb_logit_*.

This script demonstrates the use of variational Bayesian logistic regression
applied to a dataset with low-dimensional inputs, to recover the regression
coefficients, and to generate test-set predictions. Its performance is
compared to linear discriminant analysis.

Copyright (c) 2014-2019, Jan Drugowitsch (original MATLAB code)
All rights reserved.
See the file LICENSE for licensing information.
"""

import numpy as np

from _helpers import finish, fisher_lda_with_bias, parse_args, style_axes
from vblinlogit import vb_logit_fit, vb_logit_fit_iter, vb_logit_pred


def main(save_dir=None, show=True, seed=0):
    import matplotlib.pyplot as plt

    # set RNG seed and plot limits
    rng = np.random.default_rng(seed)
    wlims = (-2.5, 2.5)

    # settings
    D = 3            # dimensionality of input
    N = 100          # size of training set
    N_test = 1000    # size of test set

    # generate data
    w = rng.standard_normal(D)
    X_scale = 5

    def gen_X(n):
        # ensure class balance, with roughly 50% of samples obeying
        # w1 + w2 x2 + w3 x3 > 0
        x2 = X_scale * (rng.random(n) - 0.5)
        x3 = X_scale * (rng.random(n) - 0.5) - (w[0] + x2 * w[1]) / w[2]
        return np.column_stack([np.ones(n), x2, x3])

    X = gen_X(N)
    X_test = gen_X(N_test)
    # p(y)'s, and noisy y's
    py = 1 / (1 + np.exp(-X @ w))
    y = 2.0 * (rng.random(N) < py) - 1
    y_test = 2.0 * (rng.random(N_test) < 1 / (1 + np.exp(-X_test @ w))) - 1

    # estimate coefficients, form predictions on train & test sets
    # VB logistic regression
    w_VB, V_VB, invV_VB = vb_logit_fit(X, y)[:3]
    py_VB = vb_logit_pred(X, w_VB, V_VB, invV_VB)
    py_test_VB = vb_logit_pred(X_test, w_VB, V_VB, invV_VB)
    # VB logistic regression without hyper-prior
    w_VB1, V_VB1, invV_VB1 = vb_logit_fit_iter(X, y)[:3]
    py_VB1 = vb_logit_pred(X, w_VB1, V_VB1, invV_VB1)
    py_test_VB1 = vb_logit_pred(X_test, w_VB1, V_VB1, invV_VB1)
    # Fisher linear discriminant analysis (LDA)
    w_LD = fisher_lda_with_bias(X, y == 1)
    y_LD = 2 * (X @ w_LD > 0) - 1
    y_test_LD = 2 * (X_test @ w_LD > 0) - 1
    # output train and test-set error
    loss = dict(
        train_LDA=np.mean(y_LD != y),
        train_VB=np.mean(2 * (py_VB > 0.5) - 1 != y),
        train_VBiter=np.mean(2 * (py_VB1 > 0.5) - 1 != y),
        test_LDA=np.mean(y_test_LD != y_test),
        test_VB=np.mean(2 * (py_test_VB > 0.5) - 1 != y_test),
        test_VBiter=np.mean(2 * (py_test_VB1 > 0.5) - 1 != y_test))
    print(f"training set 0-1 loss: LDA = {loss['train_LDA']:f}, "
          f"VB = {loss['train_VB']:f}, VBiter = {loss['train_VBiter']:f}")
    print(f"test     set 0-1 loss: LDA = {loss['test_LDA']:f}, "
          f"VB = {loss['test_VB']:f}, VBiter = {loss['test_VBiter']:f}")

    # plot true vs. estimated coefficients
    f1, ax = plt.subplots()
    ax.set_xlim(wlims)
    ax.set_ylim(wlims)
    # error bars
    pm = 1.96 * np.array([-1, 1])
    for i in range(D):
        ax.plot(w[i] * np.ones(2) + 0.02, w_VB[i] + np.sqrt(V_VB[i, i]) * pm,
                '-', lw=0.25, color=(0.8, 0.5, 0.5))
        ax.plot(w[i] * np.ones(2) - 0.02, w_VB1[i] + np.sqrt(V_VB1[i, i]) * pm,
                '-', lw=0.25, color=(0.8, 0.5, 0.8))
    # means
    h1, = ax.plot(w + 0.02, w_VB, 'o', ms=3, mfc=(0.8, 0, 0), mec='none')
    h2, = ax.plot(w - 0.02, w_VB1, 's', ms=3, mfc=(0.8, 0, 0.8), mec='none')
    h3, = ax.plot(w, w_LD, '+', ms=3, mec=(0, 0, 0.8), mew=1)
    xymin = min(ax.get_xlim()[0], ax.get_ylim()[0])
    xymax = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([xymin, xymax], [xymin, xymax], 'k--', lw=0.5)
    ax.legend([h1, h2, h3], ['VB', 'VB w/o hyperprior', 'LDA'])
    style_axes(ax)
    ax.set_xlabel('$w$')
    ax.set_ylabel('$w_{LD}$, $w_{VB}$')

    # plot separating hyperplanes
    f2, ax = plt.subplots()
    # scatterplot of training data, colored by p(y=1)
    for i in range(N):
        m = 'o' if y[i] == 1 else '+'
        color = np.clip(np.array([0.2, 0.4, 0.8])
                        + py[i] * np.array([0.6, 0, -0.6]), 0, 1)
        ax.plot(X[i, 1], X[i, 2], m, ms=4, mfc='none', mec=color, mew=1)
    # separating hyperplanes where w1 + w2 x + w3 y = 0
    xlim = np.array(ax.get_xlim())
    h1, = ax.plot(xlim, -(w[0] + w[1] * xlim) / w[2], 'k-', lw=1)
    h2, = ax.plot(xlim, -(w_LD[0] + w_LD[1] * xlim) / w_LD[2], '-', lw=1,
                  color=(0, 0, 0.8))
    h3, = ax.plot(xlim, -(w_VB[0] + w_VB[1] * xlim) / w_VB[2], '-', lw=1,
                  color=(0.8, 0, 0))
    h4, = ax.plot(xlim, -(w_VB1[0] + w_VB1[1] * xlim) / w_VB1[2], '-', lw=1,
                  color=(0.8, 0, 0.8))
    ax.set_xlim(xlim)
    ax.legend([h1, h2, h3, h4], ['true', 'LDA', 'VB', 'VB w/o hyperprior'])
    style_axes(ax, aspect=4 / 3)
    ax.set_xlabel('$x_2$')
    ax.set_ylabel('$x_3$')

    finish([f1, f2], 'vb_logit_example_coeff', save_dir, show)
    return loss


if __name__ == '__main__':
    main(**parse_args(__doc__.splitlines()[0]))
