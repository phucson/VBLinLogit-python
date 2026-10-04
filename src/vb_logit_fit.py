import warnings

import numpy as np
from scipy.special import gammaln

from logdet import logdet


def vb_logit_fit(X, y, a0=1e-2, b0=1e-4):
    """[w, V, invV, logdetV, E_a, L] = vb_logit_fit(X, y)

    returns parpameters of a fitted logit model

    p(y = 1 | x, w) = 1 / (1 + exp(- w' * x)),

    with a shrinkage prior on w.

    The function expects the arguments
    - X: N x D matrix of training input samples, one per row
    - y: N-element column vector of corresponding output {-1, 1} samples
    - a0, b0 (optional): scalar shrinkage prior parameters
    If not given, the prior/hyper-prior parameters default to a0 = 1e-2,
    b0 = 1e-4, resulting in an weak shrinkage prior.

    It returns
    - w: posterior weight D-element mean vector
    - V: posterior weight D x D covariance matrix
    - invV, logdetV: inverse of V, and its log-determinant
    - E_a: scalar mean E(a) of shrinkage posterior
    - L: variational bound, lower-bounding the log-model evidence p(y | X)

    The underlying generative model assumes a weight vector prior

    p(w | a) = p(w | 0, a^-1 I),

    and hyperprior

    p(a) = Gam(a | a0, b0).

    The function returns the parameters of the posterior

    p(w1 | X, y) = N(w1 | w, V).

    Copyright (c) 2013-2019, Jan Drugowitsch
    All rights reserved.
    See the file LICENSE for licensing information.
    """

    ## pre-compute some constants
    N, D = X.shape
    max_iter = 500
    an = a0 + 0.5 * D;    gammaln_an_an = gammaln(an) + an
    t_w = 0.5 * np.sum(X * y[:, None], 0).T


    ## start first iteration kind of here, with xi = 0 -> lam_xi = 1/8
    lam_xi = np.ones(N) / 8
    E_a = a0 / b0
    invV = E_a * np.eye(D) + 2 * X.T @ (X * lam_xi[:, None])
    V = np.linalg.inv(invV)
    w = V @ t_w
    bn = b0 + 0.5 * (w.T @ w + np.trace(V))
    L_last = - N * np.log(2) \
             + 0.5 * (w.T @ invV @ w - logdet(invV)) \
             - an / bn * b0 - an * np.log(bn) + gammaln_an_an


    ## update xi, bn, (V, w) iteratively
    for i in range(1, max_iter + 1):
        # update xi by EM-algorithm
        xi = np.sqrt(np.sum(X * (X @ (V + np.outer(w, w))), 1))
        lam_xi = lam(xi)

        # update posterior parameters of a based on xi
        bn = b0 + 0.5 * (w.T @ w + np.trace(V))
        E_a = an / bn

        # recompute posterior parameters of w
        invV = E_a * np.eye(D) + 2 * X.T @ (X * lam_xi[:, None])
        V = np.linalg.inv(invV)
        logdetV = - logdet(invV)
        w = V @ t_w

        # variational bound, ingnoring constant terms for now
        L = - np.sum(np.log(1 + np.exp(- xi))) + np.sum(lam_xi * xi ** 2) \
            + 0.5 * (w.T @ invV @ w + logdetV - np.sum(xi)) \
            - E_a * b0 - an * np.log(bn) + gammaln_an_an

        # either stop if variational bound grows or change is < 0.001%
        # HACK ALARM: theoretically, the bound should never grow, and it doing
        # so points to numerical instabilities. As it seems, these start to
        # occur close to the optimal bound, which already points to a good
        # approximation.
        if (L_last > L) or (abs(L_last - L) < abs(0.00001 * L)):
            break
        L_last = L
    if i == max_iter:
        warnings.warn(
            'Bayesian logistic regression reached maximum number of iterations.')

    ## add constant terms to variational bound
    L = L - gammaln(a0) + a0 * np.log(b0)

    return w, V, invV, logdetV, E_a, L


def lam(xi):
    # returns 1 / (4 * xi) * tanh(xi / 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.tanh(xi / 2) / (4 * xi)
    # fix values where xi = 0
    out[np.isnan(out)] = 1/8
    return out
