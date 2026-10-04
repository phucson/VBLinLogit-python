import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from vb_linear_fit_ard import vb_linear_fit_ard


def vb_linear_fit_test():
    ## unit tests for vb_linear_fit_ard(.)

    ## settings
    wgen = np.array([1, 2, 3])  # test weight vector
    tau = 2          # test noise precision
    N_small = 50     # training set (small)
    N_large = 10000  # training set (large)
    a0 = 1e-2
    b0 = 1e-4
    c0 = 1e-2
    d0 = 1e-4
    genX = lambda N: np.linspace(0, 1, N)[:, None] ** np.arange(len(wgen))
    genY = lambda X: X @ wgen + np.sqrt(1 / tau) * np.random.randn(X.shape[0])


    ## test size of returned values
    print('Testing vb_linear_fit_ard(.)')
    print('Size of return values                      ', end='')
    X = genX(N_small)
    y = genY(X)
    w, V, invV, logdetV, an, bn, E_a, L = vb_linear_fit_ard(X, y)
    if not np.shape(w) == np.shape(wgen):
        print('ERROR: w size mismatch')
    elif not np.shape(V) == (len(wgen), len(wgen)):
        print('ERROR: V size mismatch')
    elif not np.shape(invV) == (len(wgen), len(wgen)):
        print('ERROR: invV size mismatch')
    elif not np.size(logdetV) == 1:
        print('ERROR: logdetV size mismatch')
    elif not np.size(an) == 1:
        print('ERROR: an size mismatch')
    elif not np.size(bn) == 1:
        print('ERROR: bn size mismatch')
    elif not np.shape(E_a) == np.shape(wgen):
        print('ERROR: E_a size mismatch')
    elif not np.size(L) == 1:
        print('ERROR: L size mismatch')
    else:
        print('OK')


    ## test consistency across repeated function calls
    print('Consistency across repeated function calls ', end='')
    X = genX(N_small)
    y = genY(X)
    w1, V1, invV1, logdetV1, an1, bn1, E_a1, L1 = vb_linear_fit_ard(X, y)
    w2, V2, invV2, logdetV2, an2, bn2, E_a2, L2 = vb_linear_fit_ard(X, y)
    if np.linalg.norm(w1 - w2) > 1e-10:
        print('ERROR: w mismatch')
    elif np.linalg.norm(V1 - V2) > 1e-10:
        print('ERROR: V mismatch')
    elif np.linalg.norm(invV1 - invV2) > 1e-10:
        print('ERROR: invV mismatch')
    elif abs(logdetV1 - logdetV2) > 1e-10:
        print('ERROR: logdetV mismatch')
    elif abs(an1 - an2) > 1e-10:
        print('ERROR: an mismatch')
    elif abs(bn1 - bn2) > 1e-10:
        print('ERROR: bn mismatch')
    elif np.linalg.norm(E_a1 - E_a2) > 1e-10:
        print('ERROR: E_a mismatch')
    elif abs(L1 - L2) > 1e-10:
        print('ERROR: L mismatch')
    else:
        print('OK')


    ## test optional arguments
    print('Use of optional arguments                  ', end='')
    X = genX(N_small)
    y = genY(X)
    w1, V1 = vb_linear_fit_ard(X, y)[:2]
    w2, V2 = vb_linear_fit_ard(X, y, a0)[:2]
    w3, V3 = vb_linear_fit_ard(X, y, a0, b0)[:2]
    w4, V4 = vb_linear_fit_ard(X, y, a0, b0, c0)[:2]
    w5, V5 = vb_linear_fit_ard(X, y, a0, b0, c0, d0)[:2]
    if np.linalg.norm(w1 - w2) > 1e-10 or np.linalg.norm(V1 - V2) > 1e-10:
        print('ERROR: mismatch with a0 argument')
    elif np.linalg.norm(w1 - w3) > 1e-10 or np.linalg.norm(V1 - V3) > 1e-10:
        print('ERROR: mismatch with b0 argument')
    elif np.linalg.norm(w1 - w4) > 1e-10 or np.linalg.norm(V1 - V4) > 1e-10:
        print('ERROR: mismatch with b0 argument')
    elif np.linalg.norm(w1 - w5) > 1e-10 or np.linalg.norm(V1 - V5) > 1e-10:
        print('ERROR: mismatch with b0 argument')
    else:
        print('OK')


    ## test weight estimates
    print('Weight estimates                           ', end='')
    X = genX(N_large)
    y = genY(X)
    w = vb_linear_fit_ard(X, y)[0]
    if np.linalg.norm(w - wgen) > 0.5:
        print('ERROR: ||w -  wtrue|| > 0.5')
    else:
        print('OK')


if __name__ == '__main__':
    vb_linear_fit_test()
