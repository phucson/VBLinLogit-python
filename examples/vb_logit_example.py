"""Simple script demonstrating the use of vb_logit_fit and vb_logit_fit_ard.

This script demonstrates the use of variational Bayesian logistic
regression without and with automated relevance determination (ARD). It
generates two datasets. The first noisily maps a d-dimensional input to a
categorical output. The second does the same, while adding d_extra
additional uninformative dimensions to the input.

The script compares multiple linear regression approaches on these
datasets:
- Fisher linear discriminant analysis
- Variational Bayesian logistic regression
- Variational Bayesian logistic regression with Automated Relevance
  Determination (ARD)
Linear discriminant analysis expected to overfit, in particular in the
presence of uninformative dimensions. The method with ARD is expected to be
most robust against addition of such uninformative dimensions. Note that,
depending on the specifics of the noise, variational Bayes might not always
outperform linear discriminant analysis.

Copyright (c) 2013-2019, Jan Drugowitsch (original MATLAB code)
All rights reserved.
See the file LICENSE for licensing information.
"""

import numpy as np

from _helpers import finish, fisher_lda_with_bias, parse_args
from vblinlogit import vb_logit_fit, vb_logit_fit_ard, vb_logit_pred


def main(save_dir=None, show=True, seed=0):
    import matplotlib.pyplot as plt

    # set RNG seed
    rng = np.random.default_rng(seed)

    # dimensionality, number of data points
    d = 3            # base dimensionality of X -> y mapping
    d_extra = 10     # additional, uninformative dimensions
    N = 100          # number of data points in training set
    N_cv = 100       # number of data points in test set

    # random weight vector & predictions
    w = rng.standard_normal(d)
    # inputs for train/test set
    X = np.hstack([np.ones((N, 1)), rng.standard_normal((N, d - 1))])
    X_ext = np.hstack([X, rng.standard_normal((N, d_extra))])
    X_cv = np.hstack([np.ones((N_cv, 1)),
                      rng.standard_normal((N_cv, d + d_extra - 1))])
    # corresponding noise-free and noisy outputs
    p_y = 1 / (1 + np.exp(-X @ w))
    y = 2.0 * (rng.random(N) < p_y) - 1
    y1 = y == 1
    y_cv = 2.0 * (rng.random(N_cv) < 1 / (1 + np.exp(-X_cv[:, :d] @ w))) - 1

    # Fisher Linear Discriminant Analysis (LDA)
    # this approach finds the best linear separator of the two classes
    # without applying any regularization. It is here applied to the base and
    # extended inputs.
    w_LD = fisher_lda_with_bias(X, y1)
    w_LD_ext = fisher_lda_with_bias(X_ext, y1)
    # LDA predictions
    y_LD = 2 * (X_ext @ w_LD_ext > 0) - 1
    y_LD_cv = 2 * (X_cv @ w_LD_ext > 0) - 1

    # weights and predictions by variational Bayes (only extended input space)
    w_vb, V_vb, invV_vb = vb_logit_fit(X_ext, y)[:3]
    # posterior probabilities, and associated choices, based on p > 0.5
    p_y_vb = vb_logit_pred(X_ext, w_vb, V_vb, invV_vb)
    y_vb = 2 * (p_y_vb > 0.5) - 1
    p_y_vb_cv = vb_logit_pred(X_cv, w_vb, V_vb, invV_vb)
    y_vb_cv = 2 * (p_y_vb_cv > 0.5) - 1

    # same with ARD (only extended input space)
    w_vb_ard, V_vb, invV_vb, logdetV_vb, E_a_vb, L_vb = \
        vb_logit_fit_ard(X_ext, y)
    # posterior probabilities, and associated choices, based on p > 0.5
    p_y_vb_ard = vb_logit_pred(X_ext, w_vb_ard, V_vb, invV_vb)
    y_vb_ard = 2 * (p_y_vb_ard > 0.5) - 1
    p_y_vb_ard_cv = vb_logit_pred(X_cv, w_vb_ard, V_vb, invV_vb)
    y_vb_ard_cv = 2 * (p_y_vb_ard_cv > 0.5) - 1

    # plot data and discriminating hyperplane
    f1, ax = plt.subplots()
    ax.plot(X[~y1, 1], X[~y1, 2], 'b+', label='y=0')
    ax.plot(X[y1, 1], X[y1, 2], 'r+', label='y=1')
    xlims = np.array(ax.get_xlim())
    # discrimination at w(1) + x * w(2) + y * w(3) = 0
    lines = [(w, 'k-', 'true'), (w_LD, 'g-', 'LD'),
             (w_LD_ext, 'g--', 'LD ext'), (w_vb, 'r-', 'vb ext'),
             (w_vb_ard, 'b-', 'vb ext ard')]
    for wi, style, label in lines:
        ax.plot(xlims, -(wi[0] + wi[1] * xlims) / wi[2], style, label=label)
    ax.set_xlim(xlims)
    ax.legend()
    ax.tick_params(direction='out')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # print training set and cross-validated MAE
    mae = {
        'LD': (np.mean(np.abs(0.5 * (y - y_LD))),
               np.mean(np.abs(0.5 * (y_cv - y_LD_cv)))),
        'VB': (np.mean(np.abs(0.5 * (y - y_vb))),
               np.mean(np.abs(0.5 * (y_cv - y_vb_cv)))),
        'VB (ARD)': (np.mean(np.abs(0.5 * (y - y_vb_ard))),
                     np.mean(np.abs(0.5 * (y_cv - y_vb_ard_cv)))),
    }
    print('MAEs:       training set     test set')
    for name, (train, test) in mae.items():
        print(f'{name:<12}{train:7.5f}          {test:7.5f}')

    finish([f1], 'vb_logit_example', save_dir, show)
    return mae


if __name__ == '__main__':
    main(**parse_args(__doc__.splitlines()[0]))
