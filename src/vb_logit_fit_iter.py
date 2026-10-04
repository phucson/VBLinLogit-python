import warnings

import numpy as np


def vb_logit_fit_iter(X, y):
    """[w, V, invV, logdetV] = vb_logit_fit_iter(X, y)

    returns parpameters of a fitted logit model

    p(y = 1 | x, w) = 1 / (1 + exp(- w' * x)).

    The function expects the arguments
    - X: N x D matrix of training input samples, one per row
    - y: N-element column vector of corresponding output {-1, 1} samples

    It returns
    - w: posterior weight D-element mean vector
    - V: posterior weight D x D covariance matrix
    - invV, logdetV: inverse of V, and its log-determinant
    - L: variational bound, lower-bounding the log-model evidence p(y | X)

    The underlying generative model assumes a weight vector prior

    p(w) = N(w | 0, D^-1 I),

    where D is the size of x.

    The function returns the parameters of the posterior

    p(w1 | X, y) = N(w1 | w, V).

    Compare to vb_logit_fit[_ard], this function does not use a hyperprior,
    iterates over the inputs separately rather than processing them all at
    once, and is therefore slower, but also computationally more stable as
    it avoids computing the inverse of possibly close-to-singular matrices.

    Copyright (c) 2013-2019, Jan Drugowitsch
    All rights reserved.
    See the file LICENSE for licensing information.
    """

    N, D = X.shape
    max_iter = 500

    ## (more or less) uninformative prior
    V = np.eye(D) / D
    invV = np.eye(D) * D
    logdetV = - D * np.log(D)
    w = np.zeros(D)


    ## iterate over all x separately
    for n in range(N):
        xn = X[n, :].T

        # precompute values
        Vx = V @ xn
        VxVx = np.outer(Vx, Vx)
        c = xn.T @ Vx
        xx = np.outer(xn, xn)
        t_w = invV @ w + 0.5 * y[n] * xn

        # start iteration at xi = 0, lam_xi = 1/8
        V_xi = V - VxVx / (4 + c)
        invV_xi = invV + xx / 4
        logdetV_xi = logdetV - np.log(1 + c / 4)
        w = V_xi @ t_w

        L_last = 0.5 * (logdetV_xi + w.T @ invV_xi @ w) - np.log(2)

        for i in range(1, max_iter + 1):
            # update xi by EM algorithm
            xi = np.sqrt(xn.T @ (V_xi + np.outer(w, w)) @ xn)
            lam_xi = lam(xi)

            # Sherman-Morrison formula and Matrix determinant lemma
            V_xi = V - (2 * lam_xi / (1 + 2 * lam_xi * c)) * VxVx
            invV_xi = invV + 2 * lam_xi * xx
            logdetV_xi = logdetV - np.log(1 + 2 * lam_xi * c)
            w = V_xi @ t_w

            L = 0.5 * (logdetV_xi + w.T @ invV_xi @ w - xi) \
                - np.log(1 + np.exp(- xi)) + lam_xi * xi ** 2

            # variational bound must grow!
            if L_last > L:
                print('Last bound %6.6f, current bound %6.6f' % (L_last, L))
                raise RuntimeError('Variational bound should not reduce')
            # stop if change in variation bound is < 0.001%
            if abs(L_last - L) < abs(0.00001 * L):
                break
            L_last = L
        if i == max_iter:
            warnings.warn(
                'Bayesian logistic regression reached maximum number of iterations.')

        V = V_xi
        invV = invV_xi
        logdetV = logdetV_xi

    return w, V, invV, logdetV


def lam(xi):
    # returns 1 / (4 * xi) * tanh(xi / 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.array(np.tanh(xi / 2) / (4 * xi))
    # fix values where xi = 0
    out[np.isnan(out)] = 1/8
    return out[()]
