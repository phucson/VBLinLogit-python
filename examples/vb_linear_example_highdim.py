## high-dimensional linear regression example for vb_linear_*
#
# This script demonstrates the use of variational Bayesian linear
# regression on a high-dimensional dataset with little training data. In
# this case, the Bayesian shrinkage regularization should outperform
# maximum likelihood estimates without shrinkage. The generated dataset has
# ~1.5 training examples per dimension of the input.
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
from vb_linear_pred import vb_linear_pred


## set RNG seed and plot limits
np.random.seed(0)
wlims = [-5, 5]
ylims = [-11, 11]


## settings
D = 100             # dimensionality of the input
N = 150             # size of the training set
N_test = 50         # size of the test set

# create data
w = np.random.randn(D)
X = np.random.rand(N, D) - 0.5
X_test = np.random.rand(N_test, D) - 0.5
y = X @ w + np.random.randn(N)
y_test = X_test @ w + np.random.randn(N_test)


## preform regression and make predictions
# variational bayes linear regression, performance on train & test set
w_VB, V_VB, _, _, an_VB, bn_VB = vb_linear_fit(X, y)[:6]
y_VB = vb_linear_pred(X, w_VB, V_VB, an_VB, bn_VB)[0]
y_test_VB, lam_VB, nu_VB = vb_linear_pred(X_test, w_VB, V_VB, an_VB, bn_VB)
# maximum likelihood, performance on train & test set
# (MATLAB uses regress(.) from the Statistics Toolbox if available, which
# also returns confidence intervals wint_ML; otherwise X \ y)
wint_ML = None
w_ML = np.linalg.lstsq(X, y, rcond=None)[0]
y_ML = X @ w_ML
y_test_ML = X_test @ w_ML
# output train and test set error
print('Training set MSE: ML = %f, VB = %f' %
      (np.mean((y - y_ML) ** 2), np.mean((y - y_VB) ** 2)))
print('Test     set MSE: ML = %f, VB = %f' %
      (np.mean((y_test - y_test_ML) ** 2), np.mean((y_test - y_test_VB) ** 2)))


## plot coefficient estimates
f1 = plt.figure()
plt.xlim(wlims);  plt.ylim(wlims)
# error bars
for i in range(D):
    plt.plot(w[i] * np.array([1, 1]) - 0.01, w_VB[i] + np.sqrt(V_VB[i, i]) * 1.96 * np.array([-1, 1]), '-',
             linewidth=0.25, color=[0.8, 0.5, 0.5])
    if wint_ML is not None:
        plt.plot(w[i] * np.array([1, 1]) + 0.01, wint_ML[i, :], '-',
                 linewidth=0.25, color=[0.5, 0.5, 0.8])
# means
h1, = plt.plot(w - 0.01, w_VB, 'o', markersize=3,
               markerfacecolor=[0.8, 0, 0], markeredgecolor='none')
h2, = plt.plot(w + 0.01, w_ML, '+', markersize=3,
               markerfacecolor='none', markeredgecolor=[0, 0, 0.8], markeredgewidth=1)
xymin = min(min(plt.xlim()), min(plt.ylim()));  xymax = max(max(plt.xlim()), max(plt.ylim()))
plt.plot([xymin, xymax], [xymin, xymax], 'k--', linewidth=0.5)
plt.legend([h1, h2], ['VB', 'ML'])
ax = plt.gca();  ax.spines[['top', 'right']].set_visible(False)
ax.set_box_aspect(1);  ax.tick_params(direction='out')
plt.xlabel('$w$');  plt.ylabel('$w_{ML}$, $w_{VB}$')


## plot test set predictions
f2 = plt.figure()
plt.xlim(ylims);  plt.ylim(ylims)
y_VB_sd = np.sqrt(nu_VB / (lam_VB * (nu_VB - 2)))
for i in range(N_test):
    plt.plot(y_test[i] * np.array([1, 1]), y_test_VB[i] + y_VB_sd[i] * 1.96 * np.array([-1, 1]), '-',
             linewidth=0.25, color=[0.8, 0.5, 0.5])
h1, = plt.plot(y_test, y_test_VB, 'o', markersize=3,
               markerfacecolor=[0.8, 0, 0], markeredgecolor='none')
h2, = plt.plot(y_test, y_test_ML, '+', markersize=3,
               markerfacecolor='none', markeredgecolor=[0, 0, 0.8], markeredgewidth=1)
xymin = min(min(plt.xlim()), min(plt.ylim()));  xymax = max(max(plt.xlim()), max(plt.ylim()))
plt.plot([xymin, xymax], [xymin, xymax], 'k--', linewidth=0.5)
plt.legend([h1, h2], ['VB', 'ML'])
ax = plt.gca();  ax.spines[['top', 'right']].set_visible(False)
ax.set_box_aspect(1);  ax.tick_params(direction='out')
plt.xlabel('$y$');  plt.ylabel('$y_{ML}$, $y_{VB}$')

if __name__ == '__main__':
    plt.show()
