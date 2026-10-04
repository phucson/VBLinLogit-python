## variational Bayesian logistic regression model selection, using vb_logit_*
#
# This script demonstrates how to use the variational bound to compare
# logistic models of different complexity and choose the most adequate
# model for the given dataset. The script models a quadratic, noisy input
# -> output mapping by polynomials of increasing order. It then selects the
# order that yields the highest variational bound - a proxy for the highest
# Bayesian model evidence - which trades off model complexity with the
# model's ability to capture the data.
#
# Copyright (c) 2014-2019, Jan Drugowitsch
# All rights reserved.
# See the file LICENSE for licensing information.

import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from vb_logit_fit import vb_logit_fit
from vb_logit_pred import vb_logit_pred


## set RNG seed
np.random.seed(41)


## settings
D = 3               # 'true' polynomial order (D-1)
N = 50              # size of training set
D_LD = 6            # polynomial order used for LDA fit
Ds = np.arange(1, 11)  # order to test VB regression on
x_range = [-5, 5]

# generate data
w = np.random.randn(D)

x = x_range[0] + (x_range[1] - x_range[0]) * np.random.rand(N)
x_test = np.linspace(x_range[0], x_range[1], 300)
gen_X = lambda x, d: x[:, None] ** np.arange(d)  # (d-1)'th order polynomial
X = gen_X(x, D)
py = 1 / (1 + np.exp(- X @ w))
y = 2 * (np.random.rand(N) < py) - 1
py_test = 1 / (1 + np.exp(- gen_X(x_test, D) @ w))
y_test = 2 * (np.random.rand(len(py_test)) < py_test) - 1


## perform model selection
Ls = np.full(len(Ds), np.nan)
pred_loss = np.full((len(Ds), 2), np.nan)
for i in range(len(Ds)):
    # avoid warnings for overspecified models
    warnings.filterwarnings('ignore', 'Bayesian logistic regression reached maximum number of iterations.')
    w, V, invV, _, _, Ls[i] = vb_logit_fit(gen_X(x, Ds[i]), y)
    warnings.filterwarnings('default', 'Bayesian logistic regression reached maximum number of iterations.')
    y_pred = 2 * (vb_logit_pred(gen_X(x, Ds[i]), w, V, invV) > 0.5) - 1
    y_test_pred = \
        2 * (vb_logit_pred(gen_X(x_test, Ds[i]), w, V, invV) > 0.5) - 1
    pred_loss[i, :] = [np.mean(y_pred != y), np.mean(y_test_pred != y_test)]
i = np.argmax(Ls)
D_best = Ds[i]
print('D_best = %d' % D_best)


## predictions for selected model
# variational bayes
X_VB = gen_X(x, D_best)
w_VB, V_VB, invV_VB = vb_logit_fit(X_VB, y)[:3]
py_VB = vb_logit_pred(X_VB, w_VB, V_VB, invV_VB)
py_test_VB = vb_logit_pred(gen_X(x_test, D_best), w_VB, V_VB, invV_VB)
# Linear Fisher Discriminant Analysis
y1 = y == 1
X_LD = gen_X(x, D_LD)
w_LD = np.full(D_LD, np.nan)
w_LD[1:] = np.linalg.solve(np.cov(X_LD[y1, 1:], rowvar=False) + np.cov(X_LD[~y1, 1:], rowvar=False),
                           np.mean(X_LD[y1, 1:], 0).T - np.mean(X_LD[~y1, 1:], 0).T)
w_LD[0] = - 0.5 * (np.mean(X_LD[y1, 1:], 0) + np.mean(X_LD[~y1, 1:], 0)) @ w_LD[1:]
y_LD = 2 * (X_LD @ w_LD > 0) - 1
y_test_LD = 2 * (gen_X(x_test, D_LD) @ w_LD > 0) - 1
# output train & test prediction error
print('Training set MSE, LDA = %f, VB = %f' %
      (np.mean(y_LD != y), np.mean(2 * (py_VB > 0.5) - 1 != y)))
print('Test     set MSE, LDA = %f, VB = %f' %
      (np.mean(y_test_LD != y_test), np.mean(2 * (py_test_VB > 0.5) - 1 != y_test)))


## plot model selection result
f1 = plt.figure()
ax = [plt.gca(), None]
h1, = ax[0].plot(Ds - 1, Ls)
ax[1] = ax[0].twinx()
h2, = ax[1].plot(Ds - 1, pred_loss[:, 0])
h1.set(linestyle='-', linewidth=1, color=[0, 0, 0])
h2.set(linestyle='-', linewidth=1, color=[0.8, 0, 0])
ax[0].spines['top'].set_visible(False);  ax[1].spines['top'].set_visible(False)
ax[0].set_box_aspect(1 / (4 / 3));  ax[1].set_box_aspect(1 / (4 / 3))
ax[0].set_xlabel('polynomial order')
plt.sca(ax[0])
plt.plot(np.array([1, 1]) * (D - 1), plt.ylim(), 'k--', linewidth=0.5)
plt.ylabel('variational bound')
ax[0].tick_params(direction='out')
plt.sca(ax[1])
h3, = plt.plot(Ds - 1, pred_loss[:, 1], '--', linewidth=1, color=[0.8, 0, 0])
plt.ylabel('0-1 loss')
plt.legend([h1, h2, h3], ['vari. bound', 'train loss', 'test loss'])
ax[1].tick_params(direction='out')


## plot prediction
f2 = plt.figure();  plt.xlim(x_range);  plt.ylim([0, 1])
plt.plot(x_test, py_test, 'k-', linewidth=1)
plt.plot(x_test, py_test_VB, '--', color=[0.8, 0, 0], linewidth=1)
plt.plot(x_test, 1 / (1 + np.exp(- gen_X(x_test, D_LD) @ w_LD)), '-',
         color=[0, 0, 0.8], linewidth=1)
plt.plot(x[y1], 1 - 0.05 * np.random.rand(*y[y1].shape), '+',
         markersize=4, color=[0.2, 0.4, 0.8])
plt.plot(x[~y1], 0.05 * np.random.rand(*y[~y1].shape), 'o',
         markersize=4, color=[0.8, 0.4, 0.2], markerfacecolor='none')
plt.legend(['p(y=1)', 'VB p(y=1)', 'LDA p(y=1)', 'y=1', 'y=0'])
ax = plt.gca();  ax.spines[['top', 'right']].set_visible(False)
ax.set_box_aspect(1 / (4 / 3));  ax.tick_params(direction='out')
plt.xlabel('$x$')
plt.ylabel('$p_{true/model}(y=1)$')

if __name__ == '__main__':
    plt.show()
