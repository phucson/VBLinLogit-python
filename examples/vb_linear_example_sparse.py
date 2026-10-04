## sparse linear regression example for vb_linear_*
#
# This example demonstrates the ability of automated relevance
# determination (ARD) to detect and ignore irrelevant input dimensions. The
# script shows this by generating an input -> output mapping with a
# D-dimensional input space, of which only a small subset of D_eff
# dimensions determine the output. It then compares variational Bayesian
# linear regression without and with ARD, and a least-square estimate, and
# shows that the variant with ARD is better able to estimate the regression
# coefficients, and also provides lower-error predictions on the test set.
#
# Copyright (c) 2014-2019, Jan Drugowitsch
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


## set RNG seed and plot limits
np.random.seed(0)
wlims = [-5, 5]
ylims = [-15, 15]


## settings
D = 1000         # full input dimensionality
D_eff = 100      # number of effective input dimensions
N = 500          # number of training set examples
N_test = 50      # number of test set examples

# create data
w = np.concatenate([np.random.randn(D_eff), np.zeros(D - D_eff)])
X = np.random.rand(N, D) - 0.5
X_test = np.random.rand(N_test, D) - 0.5
y = X @ w + np.random.randn(N)
y_test = X_test @ w + np.random.randn(N_test)


## preform regression and make predictions
# variational bayes linear regression, train & test-set predictions
w_VB, V_VB, _, _, an_VB, bn_VB = vb_linear_fit(X, y)[:6]
y_VB = vb_linear_pred(X, w_VB, V_VB, an_VB, bn_VB)[0]
y_test_VB, lam_VB, nu_VB = vb_linear_pred(X_test, w_VB, V_VB, an_VB, bn_VB)
# variational bayes linear regression with ARD, train & test-set predictions
w_VB2, V_VB2, _, _, an_VB2, bn_VB2 = vb_linear_fit_ard(X, y)[:6]
y_VB2 = vb_linear_pred(X, w_VB2, V_VB2, an_VB2, bn_VB2)[0]
y_test_VB2, lam_VB2, nu_VB2 = vb_linear_pred(X_test, w_VB2, V_VB2, an_VB2, bn_VB2)
# maximum likelihood, train & test-set predictions
# (MATLAB uses regress(.) from the Statistics Toolbox if available, which
# also returns confidence intervals wint_ML; otherwise X \ y. As N < D here,
# lstsq returns the minimum-norm solution, whereas MATLAB's X \ y returns a
# basic solution with at most N non-zero elements)
wint_ML = None
w_ML = np.linalg.lstsq(X, y, rcond=None)[0]
y_ML = X @ w_ML
y_test_ML = X_test @ w_ML
# output train and test set error
print('Training set MSE: ML = %f, VB = %f, VB w/ ARD = %f' %
      (np.mean((y - y_ML) ** 2), np.mean((y - y_VB) ** 2), np.mean((y - y_VB2) ** 2)))
print('Test     set MSE: ML = %f, VB = %f, VB w/ ARD = %f' %
      (np.mean((y_test - y_test_ML) ** 2), np.mean((y_test - y_test_VB) ** 2),
       np.mean((y_test - y_test_VB2) ** 2)))


## plot coefficient estimates
f1 = plt.figure()
plt.xlim(wlims);  plt.ylim(wlims)
# error bars
for i in range(D):
    plt.plot(w[i] * np.array([1, 1]) - 0.02, w_VB[i] + np.sqrt(V_VB[i, i]) * 1.96 * np.array([-1, 1]), '-',
             linewidth=0.25, color=[0.8, 0.5, 0.5])
    plt.plot(w[i] * np.array([1, 1]) + 0.02, w_VB2[i] + np.sqrt(V_VB2[i, i]) * 1.96 * np.array([-1, 1]), '-',
             linewidth=0.25, color=[0.5, 0.8, 0.5])
    if wint_ML is not None:
        plt.plot(w[i] * np.array([1, 1]), wint_ML[i, :], '-',
                 linewidth=0.25, color=[0.5, 0.5, 0.8])
# means
h1, = plt.plot(w - 0.02, w_VB, 'o', markersize=3,
               markerfacecolor=[0.8, 0, 0], markeredgecolor='none')
h2, = plt.plot(w + 0.02, w_VB2, 's', markersize=3,
               markerfacecolor=[0, 0.8, 0], markeredgecolor='none')
h3, = plt.plot(w, w_ML, '+', markersize=3,
               markerfacecolor='none', markeredgecolor=[0, 0, 0.8], markeredgewidth=1)
xymin = min(min(plt.xlim()), min(plt.ylim()));  xymax = max(max(plt.xlim()), max(plt.ylim()))
plt.plot([xymin, xymax], [xymin, xymax], 'k--', linewidth=0.5)
plt.legend([h1, h2, h3], ['VB', 'VB w/ ARD', 'ML'])
ax = plt.gca();  ax.spines[['top', 'right']].set_visible(False)
ax.set_box_aspect(1);  ax.tick_params(direction='out')
plt.xlabel('$w$')
plt.ylabel('$w_{ML}$, $w_{VB}$')


## plot test set predictions
f2 = plt.figure()
plt.xlim(ylims);  plt.ylim(ylims)
y_VB_sd = np.sqrt((nu_VB / (nu_VB - 2)) / lam_VB)
y_VB2_sd = np.sqrt((nu_VB2 / (nu_VB2 - 2)) / lam_VB2)
# error bars
for i in range(N_test):
    plt.plot(y_test[i] * np.array([1, 1]) - 0.02, y_test_VB[i] + y_VB_sd[i] * 1.96 * np.array([-1, 1]), '-',
             linewidth=0.25, color=[0.8, 0.5, 0.5])
    plt.plot(y_test[i] * np.array([1, 1]) + 0.02, y_test_VB2[i] + y_VB2_sd[i] * 1.96 * np.array([-1, 1]), '-',
             linewidth=0.25, color=[0.5, 0.8, 0.5])
# means
h1, = plt.plot(y_test - 0.02, y_test_VB, 'o', markersize=3,
               markerfacecolor=[0.8, 0, 0], markeredgecolor='none')
h2, = plt.plot(y_test + 0.02, y_test_VB2, 's', markersize=3,
               markerfacecolor=[0, 0.8, 0], markeredgecolor='none')
h3, = plt.plot(y_test, y_test_ML, '+', markersize=3,
               markerfacecolor='none', markeredgecolor=[0, 0, 0.8], markeredgewidth=1)
xymin = min(min(plt.xlim()), min(plt.ylim()));  xymax = max(max(plt.xlim()), max(plt.ylim()))
plt.plot([xymin, xymax], [xymin, xymax], 'k--', linewidth=0.5)
plt.legend([h1, h2, h3], ['VB', 'VB w/ ARD', 'ML'])
ax = plt.gca();  ax.spines[['top', 'right']].set_visible(False)
ax.set_box_aspect(1);  ax.tick_params(direction='out')
plt.xlabel('$y$')
plt.ylabel('$y_{ML}$, $y_{VB}$')

if __name__ == '__main__':
    plt.show()
