import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from vb_linear_fit_ard import vb_linear_fit_ard
from vb_linear_pred import vb_linear_pred


def vb_linear_pred_test():
    ## unit tests for vb_linear_pred(.)

    ## settings
    wgen = np.array([1, 2, 3])  # test weight vector
    tau = 2          # test noise precision
    N_small = 50     # training set (small)
    N_large = 10000  # training set (large)
    genX = lambda N: np.linspace(0, 1, N)[:, None] ** np.arange(len(wgen))
    genY = lambda X: X @ wgen + np.sqrt(1 / tau) * np.random.randn(X.shape[0])


    ## test size of returned values
    print('Testing vb_linear_pred(.)')
    print('Size of return values                      ', end='')
    X = genX(N_small)
    y = genY(X)
    w, V, _, _, an, bn = vb_linear_fit_ard(X, y)[:6]
    mu, lambda_, nu = vb_linear_pred(X, w, V, an, bn)
    if not np.shape(mu) == (N_small,):
        print('ERROR: mu size mismatch')
    elif not np.shape(lambda_) == (N_small,):
        print('ERROR: lambda size mismatch')
    elif not np.size(nu) == 1:
        print('ERROR: nu size mismatch')
    else:
        print('OK')


    ## test consistency across repeated function calls
    print('Consistency across repeated function calls ', end='')
    X = genX(N_small)
    y = genY(X)
    w, V, _, _, an, bn = vb_linear_fit_ard(X, y)[:6]
    mu1, lambda1, nu1 = vb_linear_pred(X, w, V, an, bn)
    mu2, lambda2, nu2 = vb_linear_pred(X, w, V, an, bn)
    if np.linalg.norm(mu1 - mu2) > 1e-10:
        print('ERROR: mu mismatch')
    elif np.linalg.norm(lambda1 - lambda2) > 1e-10:
        print('ERROR: lambda mismatch')
    elif abs(nu1 - nu2) > 1e-10:
        print('ERROR: nu mismatch')
    else:
        print('OK')


    ## test output predictions
    print('Output predictions                         ', end='')
    X = genX(N_large)
    y = genY(X)
    w, V, _, _, an, bn = vb_linear_fit_ard(X, y)[:6]
    mu = vb_linear_pred(X, w, V, an, bn)[0]
    if np.mean((mu - y) ** 2) > 1:
        print('ERROR: ||yest -  ytrue|| > 1')
    else:
        print('OK')


if __name__ == '__main__':
    vb_linear_pred_test()
