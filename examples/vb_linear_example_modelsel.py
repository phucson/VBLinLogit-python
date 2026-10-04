"""Model selection with vb_linear_*.

This script demonstrates how to use the variational bound to compare linear
models of different complexity and choose the most adequate model for the
given dataset. The script models a quadratic, noisy input -> output mapping
by polynomials of increasing order. It then selects the order that yields
the highest variational bound - a proxy for the highest Bayesian model
evidence - which trades off model complexity with the model's ability to
capture the data.

Copyright (c) 2014-2019, Jan Drugowitsch (original MATLAB code)
All rights reserved.
See the file LICENSE for licensing information.
"""

import numpy as np

from _helpers import finish, parse_args, poly_basis, style_axes
from vblinlogit import vb_linear_fit, vb_linear_pred


def main(save_dir=None, show=True, seed=1):
    import matplotlib.pyplot as plt

    # set RNG seed
    rng = np.random.default_rng(seed)

    # settings
    D = 3               # 'true' polynomial order (D-1)
    N = 10              # size of training set
    D_ML = 6            # polynomial order used for maximum likelihood fit
    Ds = np.arange(1, 11)  # orders to test for VB regression
    x_range = (-5, 5)

    # generate data
    w = rng.standard_normal(D)
    x = x_range[0] + (x_range[1] - x_range[0]) * rng.random(N)
    x_test = np.linspace(x_range[0], x_range[1], 100)
    X = poly_basis(x, D)
    # output is based on (D-1)'th order polynomial
    y = X @ w + rng.standard_normal(N)
    y_test = poly_basis(x_test, D) @ w

    # perform model selection
    # this is done by fitting polynomials of different order, and comparing
    # their associated variational bound. The model with the highest bound is
    # deemed to feature the best fit.
    Ls = np.full(len(Ds), np.nan)
    for i, d in enumerate(Ds):
        Ls[i] = vb_linear_fit(poly_basis(x, d), y)[7]
    D_best = Ds[np.argmax(Ls)]
    print(f'Selected polynomial order: {D_best - 1} (true order: {D - 1})')

    # predictions for selected model
    # variational bayes, predictions for training & test set
    w_VB, V_VB, _, _, an_VB, bn_VB = vb_linear_fit(poly_basis(x, D_best), y)[:6]
    y_VB, lam_VB, nu_VB = vb_linear_pred(poly_basis(x_test, D_best),
                                         w_VB, V_VB, an_VB, bn_VB)
    y_VB_sd = np.sqrt(nu_VB / (lam_VB * (nu_VB - 2)))
    # maximum likelihood, predictions for training & test set
    w_ML = np.linalg.lstsq(poly_basis(x, D_ML), y, rcond=None)[0]
    y_ML = poly_basis(x_test, D_ML) @ w_ML
    # output test set prediction error
    mse = dict(test_ML=np.mean((y_test - y_ML) ** 2),
               test_VB=np.mean((y_test - y_VB) ** 2))
    print(f"Test set MSE, ML = {mse['test_ML']:f}, VB = {mse['test_VB']:f}")

    # plot model selection result
    f1, ax = plt.subplots()
    ax.plot(Ds - 1, Ls, 'k-', lw=1)
    ax.axvline(D - 1, color='k', ls='--', lw=0.5)
    style_axes(ax, aspect=4 / 3)
    ax.set_xlabel('polynomial order')
    ax.set_ylabel('variational bound')

    # plot prediction
    f2, ax = plt.subplots()
    # shaded CI area
    ax.fill_between(x_test, y_VB - 1.96 * y_VB_sd, y_VB + 1.96 * y_VB_sd,
                    color=(0.9, 0.9, 0.9), lw=0)
    # true and estimated outputs
    h1, = ax.plot(x_test, y_test, 'k-', lw=1)
    h2, = ax.plot(x_test, y_VB, '--', color=(0.8, 0, 0), lw=1)
    h3, = ax.plot(x_test, y_ML, '-.', color=(0, 0, 0.8), lw=1)
    h4, = ax.plot(x, y, 'k+', ms=5)
    ax.legend([h1, h2, h3, h4], ['true', 'VB', 'ML', 'data'])
    style_axes(ax, aspect=4 / 3)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$, $y_{ML}$, $y_{VB}$')

    finish([f1, f2], 'vb_linear_example_modelsel', save_dir, show)
    return dict(D_best=D_best, Ls=Ls, **mse)


if __name__ == '__main__':
    main(**parse_args(__doc__.splitlines()[0]))
