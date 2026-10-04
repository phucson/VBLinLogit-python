"""Simple script demonstrating the use of vb_linear_fit and vb_linear_fit_ard.

This script tests linear regression on two generated dataset. Both feature
linear input -> output mappings, but only noisy outputs are observed. They
differ in the dimensionality of the inputs. The first features a
d-dimensional input, whereas the second has an additional d_extra
dimensions that are uninformative about the output values.

The script compares multiple linear regression approaches on these
datasets:
- Least-squares / maximum likelihood regression
- Variational Bayesian linear regression
- Variational Bayesian linear regression with Automated Relevance
  Determination (ARD)
The maximum likelihood approach is expected to overfit, in particular in
the presence of uninformative dimensions. The method with ARD is expected
to be most robust against addition of such uninformative dimensions.

Copyright (c) 2013-2019, Jan Drugowitsch (original MATLAB code)
All rights reserved.
See the file LICENSE for licensing information.
"""

import numpy as np

from _helpers import finish, parse_args, poly_basis
from vblinlogit import vb_linear_fit, vb_linear_fit_ard, vb_linear_pred


def main(save_dir=None, show=True, seed=4):
    import matplotlib.pyplot as plt

    # set RNG seed
    rng = np.random.default_rng(seed)

    # dimensionality, number of data points, noise
    d = 4            # base dimensionality of X -> y mapping
    d_extra = 10     # additional, uninformative dimensions
    N = 50           # number of data points in training set
    N_cv = 50        # number of data points in test set
    tau = 1          # inverse variance of additive noise

    # random weight vector & predictions
    w = rng.standard_normal(d)
    # inputs for train/test set
    x = rng.random(N)
    X = poly_basis(x, d)
    X_ext = poly_basis(x, d + d_extra)
    X_cv = poly_basis(rng.random(N_cv), d + d_extra)
    # corresponding noise-free and noisy outputs
    y_no_noise = X @ w
    y = y_no_noise + np.sqrt(1 / tau) * rng.standard_normal(N)
    y_cv = X_cv[:, :d] @ w
    # inputs used to plot predictions
    x_pred = np.linspace(0, 1, 100)
    X_pred = poly_basis(x_pred, d)
    X_pred_ext = poly_basis(x_pred, d + d_extra)

    # Estimate weights by least-squares / maximum likelihood
    w_ML = np.linalg.lstsq(X, y, rcond=None)[0]
    w_ML_ext = np.linalg.lstsq(X_ext, y, rcond=None)[0]

    # weights and predictions by variational bayes (only on extended space)
    w_vb, V_vb, _, _, an_vb, bn_vb = vb_linear_fit(X_ext, y)[:6]
    y_vb, lam_vb, nu_vb = vb_linear_pred(X_pred_ext, w_vb, V_vb, an_vb, bn_vb)
    # standard deviation of Student's t posterior with parameters lam_vb and
    # nu_vb
    y_vb_sd = np.sqrt(nu_vb / (lam_vb * (nu_vb - 2)))

    # the same with ARD
    w_vb_ard, V_vb, invV_vb, logdetV_vb, an_vb, bn_vb = \
        vb_linear_fit_ard(X_ext, y)[:6]
    y_vb_ard, lam_vb, nu_vb = vb_linear_pred(X_pred_ext, w_vb_ard, V_vb,
                                             an_vb, bn_vb)
    y_vb_ard_sd = np.sqrt(nu_vb / (lam_vb * (nu_vb - 2)))

    # plot fits
    f1, ax = plt.subplots()
    lw = 1.5
    ax.plot(x_pred, X_pred @ w, 'k-', lw=lw, label='true')
    ax.plot(x, y, 'k+', label='data')
    ax.plot(x_pred, X_pred @ w_ML, 'g-', lw=lw, label='ML')
    ax.plot(x_pred, X_pred_ext @ w_ML_ext, 'g--', lw=lw, label='ML ext')
    ax.plot(x_pred, y_vb, 'r-', lw=lw, label='vb ext')
    ax.plot(x_pred, y_vb_ard, 'b-', lw=lw, label='vb ext ard')
    ax.plot(x_pred, y_vb + y_vb_sd, 'r-.', lw=lw)
    ax.plot(x_pred, y_vb - y_vb_sd, 'r-.', lw=lw)
    ax.plot(x_pred, y_vb_ard + y_vb_ard_sd, 'b-.', lw=lw)
    ax.plot(x_pred, y_vb_ard - y_vb_ard_sd, 'b-.', lw=lw)
    ax.legend()
    ax.tick_params(direction='out')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # print training set and cross-validated MSE
    mse = {
        'ML': (np.mean((y - X_ext @ w_ML_ext) ** 2),
               np.mean((y_cv - X_cv @ w_ML_ext) ** 2)),
        'VB': (np.mean((y - X_ext @ w_vb) ** 2),
               np.mean((y_cv - X_cv @ w_vb) ** 2)),
        'VB (ARD)': (np.mean((y - X_ext @ w_vb_ard) ** 2),
                     np.mean((y_cv - X_cv @ w_vb_ard) ** 2)),
    }
    print('MSEs:       training set     test set')
    for name, (train, test) in mse.items():
        print(f'{name:<12}{train:7.5f}          {test:7.5f}')

    finish([f1], 'vb_linear_example', save_dir, show)
    return mse


if __name__ == '__main__':
    main(**parse_args(__doc__.splitlines()[0]))
