import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from vb_logit_fit import vb_logit_fit
from vb_logit_pred import vb_logit_pred


def vb_logit_pred_test():
    ## unit tests for vb_logit_pred(.)

    ## settings
    wgen = np.array([1, 2, 3])  # test weight vector
    N_small = 50      # training set (small)
    N_large = 100000  # training set (large)
    genX = lambda N: np.linspace(0, 1, N)[:, None] ** np.arange(len(wgen))
    genY = lambda X: 2 * (np.random.rand(X.shape[0]) < 1 / (1 + np.exp(-X @ wgen))) - 1


    ## test size of returned values
    print('Testing vb_logit_pred(.)')
    print('Size of return values                      ', end='')
    X = genX(N_small)
    y = genY(X)
    w, V, invV = vb_logit_fit(X, y)[:3]
    py = vb_logit_pred(X, w, V, invV)
    if not np.shape(py) == (N_small,):
        print('ERROR: mu size mismatch')
    else:
        print('OK')


    ## test consistency across repeated function calls
    print('Consistency across repeated function calls ', end='')
    X = genX(N_small)
    y = genY(X)
    w, V, invV = vb_logit_fit(X, y)[:3]
    py1 = vb_logit_pred(X, w, V, invV)
    py2 = vb_logit_pred(X, w, V, invV)
    if np.linalg.norm(py1 - py2) > 1e-10:
        print('ERROR: mu mismatch')
    else:
        print('OK')


    ## test output predictions
    print('Output predictions                         ', end='')
    X = genX(N_large)
    y = genY(X)
    w, V, invV = vb_logit_fit(X, y)[:3]
    ypred = 2 * (vb_logit_pred(X, w, V, invV) > 0.5) - 1
    if np.mean(np.abs(ypred - y)) > 0.3:
        print('ERROR: ||ypred -  ytrue|| > 1')
    else:
        print('OK')


if __name__ == '__main__':
    vb_logit_pred_test()
