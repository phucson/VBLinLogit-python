import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from vb_logit_fit import vb_logit_fit


def vb_logit_fit_test():
    ## unit tests for vb_logit_fit(.)

    ## settings
    wgen = np.array([1, 2, 3])  # test weight vector
    N_small = 50      # training set (small)
    N_large = 100000  # training set (large)
    a0 = 1e-2
    b0 = 1e-4
    genX = lambda N: np.linspace(0, 1, N)[:, None] ** np.arange(len(wgen))
    genY = lambda X: 2 * (np.random.rand(X.shape[0]) < 1 / (1 + np.exp(-X @ wgen))) - 1


    ## test size of returned values
    print('Testing vb_logit_fit(.)')
    print('Size of return values                      ', end='')
    X = genX(N_small)
    y = genY(X)
    w, V, invV, logdetV, E_a, L = vb_logit_fit(X, y)
    if not np.shape(w) == np.shape(wgen):
        print('ERROR: w size mismatch')
    elif not np.shape(V) == (len(wgen), len(wgen)):
        print('ERROR: V size mismatch')
    elif not np.shape(invV) == (len(wgen), len(wgen)):
        print('ERROR: invV size mismatch')
    elif not np.size(logdetV) == 1:
        print('ERROR: logdetV size mismatch')
    elif not np.size(E_a) == 1:
        print('ERROR: E_a size mismatch')
    elif not np.size(L) == 1:
        print('ERROR: L size mismatch')
    else:
        print('OK')


    ## test consistency across repeated function calls
    print('Consistency across repeated function calls ', end='')
    X = genX(N_small)
    y = genY(X)
    w1, V1, invV1, logdetV1, E_a1, L1 = vb_logit_fit(X, y)
    w2, V2, invV2, logdetV2, E_a2, L2 = vb_logit_fit(X, y)
    if np.linalg.norm(w1 - w2) > 1e-10:
        print('ERROR: w mismatch')
    elif np.linalg.norm(V1 - V2) > 1e-10:
        print('ERROR: V mismatch')
    elif np.linalg.norm(invV1 - invV2) > 1e-10:
        print('ERROR: invV mismatch')
    elif abs(logdetV1 - logdetV2) > 1e-10:
        print('ERROR: logdetV mismatch')
    elif abs(E_a1 - E_a2) > 1e-10:
        print('ERROR: E_a mismatch')
    elif abs(L1 - L2) > 1e-10:
        print('ERROR: L mismatch')
    else:
        print('OK')


    ## test optional arguments
    print('Use of optional arguments                  ', end='')
    X = genX(N_small)
    y = genY(X)
    w1, V1 = vb_logit_fit(X, y)[:2]
    w2, V2 = vb_logit_fit(X, y, a0)[:2]
    w3, V3 = vb_logit_fit(X, y, a0, b0)[:2]
    if np.linalg.norm(w1 - w2) > 1e-10 or np.linalg.norm(V1 - V2) > 1e-10:
        print('ERROR: mismatch with a0 argument')
    elif np.linalg.norm(w1 - w3) > 1e-10 or np.linalg.norm(V1 - V3) > 1e-10:
        print('ERROR: mismatch with b0 argument')
    else:
        print('OK')


    ## test weight estimates
    print('Weight estimates                           ', end='')
    X = genX(N_large)
    y = genY(X)
    w = vb_logit_fit(X, y)[0]
    if np.linalg.norm(w - wgen) > 1:
        print('ERROR: ||w -  wtrue|| > 1')
    else:
        print('OK')


if __name__ == '__main__':
    vb_logit_fit_test()
