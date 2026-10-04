## simple script demonstrating the use of bayes_logit_fit.py and
# bayes_logit_fit_ard.py
#
# This script demonstrates the use of variational Bayesian logistic
# regression without and with automated relevance determination (ARD). It
# generates two datasets. The first noisily maps a d-dimensional input to a
# categorical output. The second does the same, while adding d_extra
# additional uninformative dimensions to the input.
#
# The script compares multiple linear regression approaches on these
# datasets:
# - Fisher linear discriminant analysis
# - Variational Bayesian logistic regression
# - Variational Bayesian logistic regression with Automated Relevance
#   Determination (ARD)
# Linear discriminant analysis expected to overfit, in particular in the
# presence of uninformative dimensions. The method with ARD is expected to
# be most robust against addition of such uninformative dimensions. Note
# that, depending on the specifics of the noise, variational Bayes might
# not always outperform linear discriminant analysis.
#
# Copyright (c) 2013-2019, Jan Drugowitsch
# All rights reserved.
# See the file LICENSE for licensing information.

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from vb_logit_fit import vb_logit_fit
from vb_logit_fit_ard import vb_logit_fit_ard
from vb_logit_pred import vb_logit_pred


## set RNG seed
np.random.seed(0)


## dimensionality, number of data points
d = 3            # base dimensionality of X -> y mapping
d_extra = 10     # additional, uninformative dimensions
N = 100          # number of data points in training set
N_cv = 100       # number of data points in test set


## random weight vector & predictions
w = np.random.randn(d)
# inputs for train/test set
X = np.hstack([np.ones((N, 1)), np.random.randn(N, d - 1)])
X_ext = np.hstack([X, np.random.randn(N, d_extra)])
X_cv = np.hstack([np.ones((N_cv, 1)), np.random.randn(N_cv, d + d_extra - 1)])
# corresponding noise-free and noisy outputs
y_no_noise = 2 * (X @ w > 0) - 1
p_y = 1 / (1 + np.exp(- X @ w))
y = 2 * (np.random.rand(N) < p_y) - 1
y1 = (y == 1)
y_cv = 2 * (np.random.rand(N_cv) < 1 / (1 + np.exp(- X_cv[:, :d] @ w))) - 1


## Fisher Linear Discriminant Analysis (LDA)
# this approach finds the best linear separator of the two classes without
# applying any regularization. It is here applied to the base and extended
# inputs.
w_LD = np.full(d, np.nan)
w_LD[1:] = np.linalg.solve(np.cov(X[~y1, 1:], rowvar=False) + np.cov(X[y1, 1:], rowvar=False),
                           np.mean(X[y1, 1:], 0).T - np.mean(X[~y1, 1:], 0).T)
w_LD[0] = - 0.5 * (np.mean(X[y1, 1:], 0) + np.mean(X[~y1, 1:], 0)) @ w_LD[1:]
w_LD_ext = np.full(d + d_extra, np.nan)
w_LD_ext[1:] = np.linalg.solve(np.cov(X_ext[~y1, 1:], rowvar=False) + np.cov(X_ext[y1, 1:], rowvar=False),
                               np.mean(X_ext[y1, 1:], 0).T - np.mean(X_ext[~y1, 1:], 0).T)
w_LD_ext[0] = - 0.5 * (np.mean(X_ext[y1, 1:], 0) + np.mean(X_ext[~y1, 1:], 0)) @ w_LD_ext[1:]
# LDA predictions
y_LD = 2 * (X_ext @ w_LD_ext > 0) - 1
y_LD_cv = 2 * (X_cv @ w_LD_ext > 0) - 1


## weights and predictions by variational Bayes (only extended input space)
w_vb, V_vb, invV_vb, _, _, _ = vb_logit_fit(X_ext, y)
# posterior probabilities, and associated choices, based on p > 0.5
p_y_vb = vb_logit_pred(X_ext, w_vb, V_vb, invV_vb)
y_vb = 2 * (p_y_vb > 0.5) - 1
p_y_vb_cv = vb_logit_pred(X_cv, w_vb, V_vb, invV_vb)
y_vb_cv = 2 * (p_y_vb_cv > 0.5) - 1

# same with ARD (only extended input space)
w_vb_ard, V_vb, invV_vb, logdetV_vb, E_a_vb, L_vb = vb_logit_fit_ard(X_ext, y)
# posterior probabilities, and associated choices, based on p > 0.5
p_y_vb_ard = vb_logit_pred(X_ext, w_vb_ard, V_vb, invV_vb)
y_vb_ard = 2 * (p_y_vb_ard > 0.5) - 1
p_y_vb_ard_cv = vb_logit_pred(X_cv, w_vb_ard, V_vb, invV_vb)
y_vb_ard_cv = 2 * (p_y_vb_ard_cv > 0.5) - 1


## plot data and discriminating hyperplane
f1 = plt.figure()
plt.plot(X[~y1, 1], X[~y1, 2], 'b+')
plt.plot(X[y1, 1], X[y1, 2], 'r+')
xlims = np.array(plt.gca().get_xlim())
# discrimination at w(1) + x * w(2) + y * w(3) = 0
plt.plot(xlims, -(w[0] + w[1] * xlims) / w[2], 'k-')
plt.plot(xlims, -(w_LD[0] + w_LD[1] * xlims) / w_LD[2], 'g-')
plt.plot(xlims, -(w_LD_ext[0] + w_LD_ext[1] * xlims) / w_LD_ext[2], 'g--')
plt.plot(xlims, -(w_vb[0] + w_vb[1] * xlims) / w_vb[2], 'r-')
plt.plot(xlims, -(w_vb_ard[0] + w_vb_ard[1] * xlims) / w_vb_ard[2], 'b-')
plt.legend(['y=0', 'y=1', 'true', 'LD', 'LD ext', 'vb ext', 'vb ext ard'])
plt.gca().tick_params(direction='out')
plt.xlabel('x')
plt.ylabel('y')

# print training set and cross-validated MAE
print('MAEs:       training set     test set')
print('LD          %7.5f          %7.5f' %
      (np.mean(np.abs(0.5 * (y - y_LD))), np.mean(np.abs(0.5 * (y_cv - y_LD_cv)))))
print('VB          %7.5f          %7.5f' %
      (np.mean(np.abs(0.5 * (y - y_vb))), np.mean(np.abs(0.5 * (y_cv - y_vb_cv)))))
print('VB (ARD)    %7.5f          %7.5f' %
      (np.mean(np.abs(0.5 * (y - y_vb_ard))), np.mean(np.abs(0.5 * (y_cv - y_vb_ard_cv)))))

if __name__ == '__main__':
    plt.show()
