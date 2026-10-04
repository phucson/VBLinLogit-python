## high-dimensional logitstic regression, using vb_logit_*
#
# This example demonstrates the ability of automated relevance
# determination (ARD) to detect and ignore irrelevant input dimensions for
# logistic regression. The script shows this by generating an input ->
# output mapping with a D-dimensional input space, of which only a small
# subset of D_eff dimensions determine the output class. It then compares
# variational Bayesian linear regression without and with ARD, and linear
# discriminant analysis, and shows that the variant with ARD is better able
# to estimate the regression coefficients, and also provides lower-error
# predictions on the test set.
#
# Copyright (c) 2014-2019, Jan Drugowitsch
# All rights reserved.
# See the file LICENSE for licensing information.

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from vb_logit_fit import vb_logit_fit
from vb_logit_fit_ard import vb_logit_fit_ard
from vb_logit_fit_iter import vb_logit_fit_iter
from vb_logit_pred import vb_logit_pred


## set RNG seed and plot limits
np.random.seed(0)
wlims = [-4, 4]


## settings
D = 1000         # full input dimensionality
D_eff = 100      # number of effective input dimensions
N = 2000         # number of training set examples
N_test = 10000   # number of test set examples
N_plot = 200     # number of examples to plot

# generate data
w = np.concatenate([np.random.randn(D_eff), np.zeros(D - D_eff)])
# inputs for train & test set
X = np.random.rand(N, D) - 0.5
X_test = np.random.rand(N_test, D) - 0.5
# output probabilities and samples for train & test set
py = 1 / (1 + np.exp(- X @ w))
y = 2 * (np.random.rand(N) < py) - 1
py_test = 1 / (1 + np.exp(-X_test @ w))
y_test = 2 * (np.random.rand(N_test) < py_test) - 1


## estimate coefficients, form predictions for train & test set
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
w_LD = np.linalg.solve(np.cov(X[y1, :], rowvar=False) + np.cov(X[~y1, :], rowvar=False),
                       np.mean(X[y1, :], 0).T - np.mean(X[~y1, :], 0).T)
c_LD = 0.5 * (np.mean(X[y1, :], 0) + np.mean(X[~y1, :], 0)) @ w_LD
y_LD = 2 * (X @ w_LD > c_LD) - 1
y_test_LD = 2 * (X_test @ w_LD > c_LD) - 1
# output train and test-set error
print(('training set 0-1 loss: LDA    = %f, VB        = %f\n'
       '                       VBiter = %f, VB w/ ARD = %f') %
      (np.mean(y_LD != y), np.mean(2 * (py_VB > 0.5) - 1 != y),
       np.mean(2 * (py_VB1 > 0.5) - 1 != y), np.mean(2 * (py_VB2 > 0.5) - 1 != y)))
print(('test     set 0-1 loss: LDA    = %f, VB        = %f\n'
       '                       VBiter = %f, VB w/ ARD = %f') %
      (np.mean(y_test_LD != y_test), np.mean(2 * (py_test_VB > 0.5) - 1 != y_test),
       np.mean(2 * (py_test_VB1 > 0.5) - 1 != y_test),
       np.mean(2 * (py_test_VB2 > 0.5) - 1 != y_test)))


## plot coefficient estimates
f1 = plt.figure()
plt.xlim(wlims);  plt.ylim(wlims)
# error bars
for i in range(D):
    plt.plot(w[i] * np.array([1, 1]) - 0.03, w_VB[i] + np.sqrt(V_VB[i, i]) * 1.96 * np.array([-1, 1]), '-',
             linewidth=0.25, color=[0.8, 0.5, 0.5])
    plt.plot(w[i] * np.array([1, 1]) - 0.01, w_VB1[i] + np.sqrt(V_VB1[i, i]) * 1.96 * np.array([-1, 1]), '-',
             linewidth=0.25, color=[0.8, 0.5, 0.8])
    plt.plot(w[i] * np.array([1, 1]) + 0.01, w_VB2[i] + np.sqrt(V_VB2[i, i]) * 1.96 * np.array([-1, 1]), '-',
             linewidth=0.25, color=[0.5, 0.8, 0.5])
# means
h1, = plt.plot(w - 0.03, w_VB, 'o', markersize=3,
               markerfacecolor=[0.8, 0, 0], markeredgecolor='none')
h2, = plt.plot(w - 0.01, w_VB1, 'd', markersize=3,
               markerfacecolor=[0.8, 0, 0.8], markeredgecolor='none')
h3, = plt.plot(w + 0.01, w_VB2, 's', markersize=3,
               markerfacecolor=[0, 0.8, 0], markeredgecolor='none')
h4, = plt.plot(w + 0.03, w_LD, '+', markersize=3,
               markerfacecolor='none', markeredgecolor=[0, 0, 0.8], markeredgewidth=1)
xymin = min(min(plt.xlim()), min(plt.ylim()));  xymax = max(max(plt.xlim()), max(plt.ylim()))
plt.plot([xymin, xymax], [xymin, xymax], 'k--', linewidth=0.5)
plt.legend([h1, h2, h3, h4], ['VB', 'VB w/o hyperpriors', 'VB w/ ARD', 'LDA'])
ax = plt.gca();  ax.spines[['top', 'right']].set_visible(False)
ax.set_box_aspect(1);  ax.tick_params(direction='out')
plt.xlabel('$w$')
plt.ylabel('$w_{ML}$, $w_{VB}$')


## plot test set predictions
f2 = plt.figure();  plt.xlim([0, 1]);  plt.ylim([0, 1])
# misclassification areas
plt.fill([0.5, 1, 1, 0.5], [0, 0, 0.5, 0.5], color=[0.95, 0.8, 0.8], edgecolor='none')
plt.fill([0, 0.5, 0.5, 0], [0.5, 0.5, 1, 1], color=[0.95, 0.8, 0.8], edgecolor='none')
# p(y=1) for N_plot samples of test set
h1, = plt.plot(py_test[:N_plot], py_test_VB[:N_plot], 'o', markersize=3,
               markerfacecolor=[0.8, 0, 0], markeredgecolor='none')
h2, = plt.plot(py_test[:N_plot], py_test_VB1[:N_plot], 'd', markersize=3,
               markerfacecolor=[0.8, 0, 0.8], markeredgecolor='none')
h3, = plt.plot(py_test[:N_plot], py_test_VB2[:N_plot], 's', markersize=3,
               markerfacecolor=[0, 0.8, 0], markeredgecolor='none')
plt.plot(plt.xlim(), plt.ylim(), 'k--', linewidth=0.5)
plt.legend([h1, h2, h3], ['VB', 'VB w/o hyperpriors', 'VB w/ ARD'])
ax = plt.gca();  ax.spines[['top', 'right']].set_visible(False)
ax.set_box_aspect(1);  ax.tick_params(direction='out')
plt.xlabel('$p_{true}(y = 1)$')
plt.ylabel('$p_{VB}(y = 1)$')

if __name__ == '__main__':
    plt.show()
