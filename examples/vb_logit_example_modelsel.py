"""Variational Bayesian logistic regression model selection, using vb_logit_*.

This script demonstrates how to use the variational bound to compare
logistic models of different complexity and choose the most adequate model
for the given dataset. The script models a quadratic, noisy input -> output
mapping by polynomials of increasing order. It then selects the order that
yields the highest variational bound - a proxy for the highest Bayesian
model evidence - which trades off model complexity with the model's ability
to capture the data.

Copyright (c) 2014-2019, Jan Drugowitsch (original MATLAB code)
All rights reserved.
See the file LICENSE for licensing information.
"""

import warnings

import numpy as np

from _helpers import (finish, fisher_lda_with_bias, parse_args, poly_basis,
                      style_axes)
from vblinlogit import MaxIterWarning, vb_logit_fit, vb_logit_pred


def main(save_dir=None, show=True, seed=3):
    import matplotlib.pyplot as plt

    # set RNG seed
    rng = np.random.default_rng(seed)

    # settings
    D = 3               # 'true' polynomial order (D-1)
    N = 50              # size of training set
    D_LD = 6            # polynomial order used for LDA fit
    Ds = np.arange(1, 11)  # order to test VB regression on
    x_range = (-5, 5)

    # generate data
    w = rng.standard_normal(D)
    x = x_range[0] + (x_range[1] - x_range[0]) * rng.random(N)
    x_test = np.linspace(x_range[0], x_range[1], 300)
    X = poly_basis(x, D)
    py = 1 / (1 + np.exp(-X @ w))
    y = 2.0 * (rng.random(N) < py) - 1
    py_test = 1 / (1 + np.exp(-poly_basis(x_test, D) @ w))
    y_test = 2.0 * (rng.random(len(py_test)) < py_test) - 1

    # perform model selection
    Ls = np.full(len(Ds), np.nan)
    pred_loss = np.full((len(Ds), 2), np.nan)
    for i, d in enumerate(Ds):
        with warnings.catch_warnings():
            # avoid warnings for overspecified models
            warnings.simplefilter('ignore', MaxIterWarning)
            w_d, V_d, invV_d, _, _, Ls[i] = vb_logit_fit(poly_basis(x, d), y)
        y_pred = 2 * (vb_logit_pred(poly_basis(x, d), w_d, V_d, invV_d)
                      > 0.5) - 1
        y_test_pred = 2 * (vb_logit_pred(poly_basis(x_test, d), w_d, V_d,
                                         invV_d) > 0.5) - 1
        pred_loss[i, :] = [np.mean(y_pred != y), np.mean(y_test_pred != y_test)]
    D_best = Ds[np.argmax(Ls)]
    print(f'D_best = {D_best}')

    # predictions for selected model
    # variational bayes
    X_VB = poly_basis(x, D_best)
    w_VB, V_VB, invV_VB = vb_logit_fit(X_VB, y)[:3]
    py_VB = vb_logit_pred(X_VB, w_VB, V_VB, invV_VB)
    py_test_VB = vb_logit_pred(poly_basis(x_test, D_best), w_VB, V_VB, invV_VB)
    # Linear Fisher Discriminant Analysis
    y1 = y == 1
    X_LD = poly_basis(x, D_LD)
    w_LD = fisher_lda_with_bias(X_LD, y1)
    y_LD = 2 * (X_LD @ w_LD > 0) - 1
    y_test_LD = 2 * (poly_basis(x_test, D_LD) @ w_LD > 0) - 1
    # output train & test prediction error
    loss = dict(train_LDA=np.mean(y_LD != y),
                train_VB=np.mean(2 * (py_VB > 0.5) - 1 != y),
                test_LDA=np.mean(y_test_LD != y_test),
                test_VB=np.mean(2 * (py_test_VB > 0.5) - 1 != y_test))
    print(f"Training set 0-1 loss, LDA = {loss['train_LDA']:f}, "
          f"VB = {loss['train_VB']:f}")
    print(f"Test     set 0-1 loss, LDA = {loss['test_LDA']:f}, "
          f"VB = {loss['test_VB']:f}")

    # plot model selection result
    f1, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    h1, = ax1.plot(Ds - 1, Ls, '-', lw=1, color=(0, 0, 0))
    ax1.axvline(D - 1, color='k', ls='--', lw=0.5)
    ax1.set_xlabel('polynomial order')
    ax1.set_ylabel('variational bound')
    h2, = ax2.plot(Ds - 1, pred_loss[:, 0], '-', lw=1, color=(0.8, 0, 0))
    h3, = ax2.plot(Ds - 1, pred_loss[:, 1], '--', lw=1, color=(0.8, 0, 0))
    ax2.set_ylabel('0-1 loss')
    ax2.legend([h1, h2, h3], ['vari. bound', 'train loss', 'test loss'])
    for ax in (ax1, ax2):
        ax.tick_params(direction='out')
        ax.spines['top'].set_visible(False)
        ax.set_box_aspect(3 / 4)

    # plot prediction
    f2, ax = plt.subplots()
    ax.set_xlim(x_range)
    ax.set_ylim(0, 1)
    ax.plot(x_test, py_test, 'k-', lw=1, label='p(y=1)')
    ax.plot(x_test, py_test_VB, '--', color=(0.8, 0, 0), lw=1,
            label='VB p(y=1)')
    ax.plot(x_test, 1 / (1 + np.exp(-poly_basis(x_test, D_LD) @ w_LD)), '-',
            color=(0, 0, 0.8), lw=1, label='LDA p(y=1)')
    ax.plot(x[y1], 1 - 0.05 * rng.random(np.sum(y1)), '+', ms=4,
            color=(0.2, 0.4, 0.8), label='y=1')
    ax.plot(x[~y1], 0.05 * rng.random(np.sum(~y1)), 'o', ms=4, mfc='none',
            color=(0.8, 0.4, 0.2), label='y=0')
    ax.legend()
    style_axes(ax, aspect=4 / 3)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$p_{true/model}(y=1)$')

    finish([f1, f2], 'vb_logit_example_modelsel', save_dir, show)
    return dict(D_best=D_best, Ls=Ls, pred_loss=pred_loss, **loss)


if __name__ == '__main__':
    main(**parse_args(__doc__.splitlines()[0]))
