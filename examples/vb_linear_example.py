## simple script demonstrating the use of vb_linear_fit.py and
# vb_linear_fit_ard.py
#
# This script tests linear regression on two generated dataset. Both
# feature linear input -> output mappings, but only noisy outputs are
# observed. They differ in the dimensionality of the inputs. The first
# features a d-dimensional input, whereas the second has an additional
# d_extra dimensions that are uninformative about the output values.
#
# The script compares multiple linear regression approaches on these
# datasets:
# - Least-squares / maximum likelihood regression
# - Variational Bayesian linear regression
# - Variational Bayesian linear regression with Automated Relevance
#   Determination (ARD)
# The maximum likelihood approach is expected to overfit, in particular in
# the presence of uninformative dimensions. The method with ARD is expected
# to be most robust against addition of such uninformative dimensions.
#
# Copyright (c) 2013-2019, Jan Drugowitsch
# All rights reserved.
# See the file LICENSE for licensing information.

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from vb_linear_fit import vb_linear_fit
from vb_linear_fit_ard import vb_linear_fit_ard
from vb_linear_pred import vb_linear_pred


## set RNG seed
np.random.seed(4)


## dimensionality, number of data points, noise
d = 4            # base dimensionality of X -> y mapping
d_extra = 10     # additional, uninformative dimensions
N = 50           # number of data points in training set
N_cv = 50        # number of data points in test set
tau = 1          # inverse variance of additive noise


## random weight vector & predictions
w = np.random.randn(d)
# inputs for train/test set
x = np.random.rand(N)
X = x[:, None] ** np.arange(d)
X_ext = x[:, None] ** np.arange(d + d_extra)
X_cv = np.random.rand(N_cv)[:, None] ** np.arange(d + d_extra)
# corresponding noise-free and noisy outputs
y_no_noise = X @ w
y = y_no_noise + np.sqrt(1 / tau) * np.random.randn(N)
y_cv = X_cv[:, :d] @ w
# inputs used to plot predictions
x_pred = np.linspace(0, 1, 100)
X_pred = x_pred[:, None] ** np.arange(d)
X_pred_ext = x_pred[:, None] ** np.arange(d + d_extra)


## Estimate weights by least-squares / maximum likelihood
w_ML = np.linalg.lstsq(X, y, rcond=None)[0]              # X \ y
w_ML_ext = np.linalg.lstsq(X_ext, y, rcond=None)[0]      # X_ext \ y
# predictions for training and test set on extended input space
y_ML = X_ext @ w_ML_ext
y_ML_cv = X_cv @ w_ML_ext


## weights and predictions by variational bayes (only on extended space)
w_vb, V_vb, _, _, an_vb, bn_vb = vb_linear_fit(X_ext, y)[:6]
y_vb, lam_vb, nu_vb = vb_linear_pred(X_pred_ext, w_vb, V_vb, an_vb, bn_vb)
# standard deviation of Student's t posterior with parameters lam_vb and nu_vb
y_vb_sd = np.sqrt(nu_vb / (lam_vb * (nu_vb - 2)))

# the same with ARD
w_vb_ard, V_vb, invV_vb, logdetV_vb, an_vb, bn_vb = vb_linear_fit_ard(X_ext, y)[:6]
y_vb_ard, lam_vb, nu_vb = vb_linear_pred(X_pred_ext, w_vb_ard, V_vb, an_vb, bn_vb)
# standard deviation of Student's t posterior with parameters lam_vb and nu_vb
y_vb_ard_sd = np.sqrt(nu_vb / (lam_vb * (nu_vb - 2)))


## plot fits
f1 = plt.figure();  plt.rcParams['lines.linewidth'] = 1.5
plt.plot(x_pred, X_pred @ w, 'k-')
plt.plot(x, y, 'k+')
plt.plot(x_pred, X_pred @ w_ML, 'g-')
plt.plot(x_pred, X_pred_ext @ w_ML_ext, 'g--')
plt.plot(x_pred, y_vb, 'r-')
plt.plot(x_pred, y_vb_ard, 'b-')
plt.plot(x_pred, y_vb + y_vb_sd, 'r-.')
plt.plot(x_pred, y_vb - y_vb_sd, 'r-.')
plt.plot(x_pred, y_vb_ard + y_vb_ard_sd, 'b-.')
plt.plot(x_pred, y_vb_ard - y_vb_ard_sd, 'b-.')
plt.legend(['true', 'data', 'ML', 'ML ext', 'vb ext', 'vb ext ard'])
plt.gca().tick_params(direction='out')
plt.xlabel('x')
plt.ylabel('y')

# print training set and cross-validated MSE
print('MSEs:       training set     test set')
print('ML          %7.5f          %7.5f' %
      (np.mean((y - X_ext @ w_ML_ext) ** 2), np.mean((y_cv - X_cv @ w_ML_ext) ** 2)))
print('VB          %7.5f          %7.5f' %
      (np.mean((y - X_ext @ w_vb) ** 2), np.mean((y_cv - X_cv @ w_vb) ** 2)))
print('VB (ARD)    %7.5f          %7.5f' %
      (np.mean((y - X_ext @ w_vb_ard) ** 2), np.mean((y_cv - X_cv @ w_vb_ard) ** 2)))

if __name__ == '__main__':
    plt.show()
