## model selection with vb_linear_*
#
# This script demonstrates how to use the variational bound to compare
# linear models of different complexity and choose the most adequate model
# for the given dataset. The script models a quadratic, noisy input ->
# output mapping by polynomials of increasing order. It then selects the
# order that yields the highest variational bound - a proxy for the highest
# Bayesian model evidence - which trades off model complexity with the
# model's ability to capture the data.
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


## set RNG seed
np.random.seed(1)


## settings
D = 3               # 'true' polynomial order (D-1)
N = 10              # size of training set
D_ML = 6            # polynomial order used for maximum likelihood fit
Ds = np.arange(1, 11)  # orders to test for VB regression
x_range = [-5, 5]

# generate data
w = np.random.randn(D)
x = x_range[0] + (x_range[1] - x_range[0]) * np.random.rand(N)
x_test = np.linspace(x_range[0], x_range[1], 100)
gen_X = lambda x, d: x[:, None] ** np.arange(d)  # (d-1)'th order polynomial
X = gen_X(x, D)
# output is based on (D-1)'th order polynomial
y = X @ w + np.random.randn(N)
y_test = gen_X(x_test, D) @ w


## perform model selection
# this is done by fitting polynomials of different order, and comparing their
# associated variational bound. The model with the highest bound is deemed to
# feature the best fit.
Ls = np.full(len(Ds), np.nan)
for i in range(len(Ds)):
    Ls[i] = vb_linear_fit(gen_X(x, Ds[i]), y)[7]
i = np.argmax(Ls)
D_best = Ds[i]


## predictions for selected model
# variational bayes, predictions for training & test set
w_VB, V_VB, _, _, an_VB, bn_VB = vb_linear_fit(gen_X(x, D_best), y)[:6]
y_VB, lam_VB, nu_VB = \
    vb_linear_pred(gen_X(x_test, D_best), w_VB, V_VB, an_VB, bn_VB)
y_VB_sd = np.sqrt(nu_VB / (lam_VB * (nu_VB - 2)))
# maximum likelihood, predictions for training & test set
w_ML = np.linalg.lstsq(gen_X(x, D_ML), y, rcond=None)[0]   # gen_X(x, D_ML) \ y
y_ML = gen_X(x_test, D_ML) @ w_ML
# output test set prediction error
print('Test set MSE, ML = %f, VB = %f' %
      (np.mean((y_test - y_ML) ** 2), np.mean((y_test - y_VB) ** 2)))


## plot model selection result
f1 = plt.figure()
plt.plot(Ds - 1, Ls, 'k-', linewidth=1)
plt.plot(np.array([1, 1]) * (D - 1), plt.ylim(), 'k--', linewidth=0.5)
ax = plt.gca();  ax.spines[['top', 'right']].set_visible(False)
ax.set_box_aspect(1 / (4 / 3));  ax.tick_params(direction='out')
plt.xlabel('polynomial order')
plt.ylabel('variational bound')


## plot prediction
f2 = plt.figure()
# shaded CI area
plt.fill(np.concatenate([x_test, np.flipud(x_test)]),
         np.concatenate([y_VB + 1.96 * y_VB_sd, np.flipud(y_VB - 1.96 * y_VB_sd)]),
         color=np.array([1, 1, 1]) * 0.9, edgecolor='none')
# true and esimtated outputs
h1, = plt.plot(x_test, y_test, 'k-', linewidth=1)
h2, = plt.plot(x_test, y_VB, '--', color=[0.8, 0, 0], linewidth=1)
h3, = plt.plot(x_test, y_ML, '-.', color=[0, 0, 0.8], linewidth=1)
h4, = plt.plot(x, y, 'k+', markersize=5)
plt.legend([h1, h2, h3, h4], ['true', 'VB', 'ML', 'data'])
ax = plt.gca();  ax.spines[['top', 'right']].set_visible(False)
ax.set_box_aspect(1 / (4 / 3));  ax.tick_params(direction='out')
plt.xlabel('$x$')
plt.ylabel('$y$, $y_{ML}$, $y_{VB}$')

if __name__ == '__main__':
    plt.show()
