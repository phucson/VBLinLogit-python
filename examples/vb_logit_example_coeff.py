## estimating coefficients and separating hyperplane, using vb_logit_*
#
# This script demonstrates the use of variational Bayesian logistic
# regression applied to a dataset with low-dimensional inputs, to recover
# the regression coefficients, and to generate test-set predictions. Its
# performance is compared to linear disciminant analysis.
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
from vb_logit_fit_iter import vb_logit_fit_iter
from vb_logit_pred import vb_logit_pred


## set RNG seed and plot limits
np.random.seed(0)
wlims = [-2.5, 2.5]


## settings
D = 3            # dimensionality of input
N = 100          # size of training set
N_test = 1000    # size of test set

# generate data
w = np.random.randn(D)
X_scale = 5
# ensure class balance, with roughly 50% of samples obeying
# w1 + w2 x2 + w3 x3 > 0
X = np.column_stack([np.ones(N), (X_scale * (np.random.rand(N) - 0.5))])
X = np.column_stack([X, (X_scale * (np.random.rand(N) - 0.5) - (w[0] + X[:, 1] * w[1]) / w[2])])
X_test = np.column_stack([np.ones(N_test), (X_scale * (np.random.rand(N_test) - 0.5))])
X_test = np.column_stack([X_test, (X_scale * (np.random.rand(N_test) - 0.5) - (w[0] + X_test[:, 1] * w[1]) / w[2])])
# p(y)'s, and noisy y's
py = 1 / (1 + np.exp(- X @ w))
y = 2 * (np.random.rand(N) < py) - 1
y_test = 2 * (np.random.rand(N_test) < 1 / (1 + np.exp(- X_test @ w))) - 1


## estimate coefficients, form predictions on train & test sets
# VB logistic regression
w_VB, V_VB, invV_VB = vb_logit_fit(X, y)[:3]
py_VB = vb_logit_pred(X, w_VB, V_VB, invV_VB)
py_test_VB = vb_logit_pred(X_test, w_VB, V_VB, invV_VB)
# VB logistic regression without hyper-prior
w_VB1, V_VB1, invV_VB1 = vb_logit_fit_iter(X, y)[:3]
py_VB1 = vb_logit_pred(X, w_VB1, V_VB1, invV_VB1)
py_test_VB1 = vb_logit_pred(X_test, w_VB1, V_VB1, invV_VB1)
# Fisher linear discriminant analysis (LDA)
y1 = y == 1
w_LD = np.full(D, np.nan)
w_LD[1:] = np.linalg.solve(np.cov(X[y1, 1:], rowvar=False) + np.cov(X[~y1, 1:], rowvar=False),
                           np.mean(X[y1, 1:], 0).T - np.mean(X[~y1, 1:], 0).T)
w_LD[0] = - 0.5 * (np.mean(X[y1, 1:], 0) + np.mean(X[~y1, 1:], 0)) @ w_LD[1:]
y_LD = 2 * (X @ w_LD > 0) - 1
y_test_LD = 2 * (X_test @ w_LD > 0) - 1
# output train and test-set error
print('training set 0-1 loss: LDA = %f, VB = %f, VBiter = %f' %
      (np.mean(y_LD != y), np.mean(2 * (py_VB > 0.5) - 1 != y),
       np.mean(2 * (py_VB1 > 0.5) - 1 != y)))
print('test     set 0-1 loss: LDA = %f, VB = %f, VBiter = %f' %
      (np.mean(y_test_LD != y_test), np.mean(2 * (py_test_VB > 0.5) - 1 != y_test),
       np.mean(2 * (py_test_VB1 > 0.5) - 1 != y_test)))


## plot true vs. estimated coefficients
f1 = plt.figure()
plt.xlim(wlims);  plt.ylim(wlims)
# error bars
for i in range(D):
    plt.plot(w[i] * np.array([1, 1]) + 0.02, w_VB[i] + np.sqrt(V_VB[i, i]) * 1.96 * np.array([-1, 1]), '-',
             linewidth=0.25, color=[0.8, 0.5, 0.5])
    plt.plot(w[i] * np.array([1, 1]) - 0.02, w_VB1[i] + np.sqrt(V_VB1[i, i]) * 1.96 * np.array([-1, 1]), '-',
             linewidth=0.25, color=[0.8, 0.5, 0.8])
# means
h1, = plt.plot(w + 0.02, w_VB, 'o', markersize=3,
               markerfacecolor=[0.8, 0, 0], markeredgecolor='none')
h2, = plt.plot(w - 0.02, w_VB1, 's', markersize=3,
               markerfacecolor=[0.8, 0, 0.8], markeredgecolor='none')
h3, = plt.plot(w, w_LD, '+', markersize=3,
               markerfacecolor='none', markeredgecolor=[0, 0, 0.8], markeredgewidth=1)
xymin = min(min(plt.xlim()), min(plt.ylim()));  xymax = max(max(plt.xlim()), max(plt.ylim()))
plt.plot([xymin, xymax], [xymin, xymax], 'k--', linewidth=0.5)
plt.legend([h1, h2, h3], ['VB', 'VB w/o hyperprior', 'LDA'])
ax = plt.gca();  ax.spines[['top', 'right']].set_visible(False)
ax.set_box_aspect(1);  ax.tick_params(direction='out')
plt.xlabel('$w$')
plt.ylabel('$w_{LD}$, $w_{VB}$')


## plot separating hyperplanes
f2 = plt.figure()
# scatterplot of training data, colored by p(y=1)
for i in range(N):
    if y[i] == 1:
        m = 'o'
    else:
        m = '+'
    plt.plot(X[i, 1], X[i, 2], m, markersize=3, markerfacecolor='none',
             markeredgecolor=np.array([0.2, 0.4, 0.8]) + py[i] * np.array([0.6, 0, -0.6]),
             markeredgewidth=0.5)
# separating hyperplanes where w1 + w2 x + w3 y = 0
xlim = np.array(plt.xlim())
h1, = plt.plot(xlim, -(w[0] + w[1] * xlim) / w[2], 'k-', linewidth=1)
h2, = plt.plot(xlim, -(w_LD[0] + w_LD[1] * xlim) / w_LD[2], '-',
               linewidth=1, color=[0, 0, 0.8])
h3, = plt.plot(xlim, -(w_VB[0] + w_VB[1] * xlim) / w_VB[2], '-',
               linewidth=1, color=[0.8, 0, 0])
h4, = plt.plot(xlim, -(w_VB1[0] + w_VB1[1] * xlim) / w_VB1[2], '-',
               linewidth=1, color=[0.8, 0, 0.8])
plt.legend([h1, h2, h3, h4], ['true', 'LDA', 'VB', 'VB w/o hyperprior'])
ax = plt.gca();  ax.spines[['top', 'right']].set_visible(False)
ax.set_box_aspect(1 / (4 / 3));  ax.tick_params(direction='out')
plt.xlabel('$x_2$');  plt.ylabel('$x_3$')

if __name__ == '__main__':
    plt.show()
