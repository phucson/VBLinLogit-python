"""Variational Bayesian logistic regression.

Python port of ``vb_logit_fit.m``, ``vb_logit_fit_ard.m``,
``vb_logit_fit_iter.m``, ``vb_logit_pred.m`` and ``vb_logit_pred_incr.m``.

Copyright (c) 2013-2019, Jan Drugowitsch (original MATLAB code)
All rights reserved.
See the file LICENSE for licensing information.
"""

import warnings

import numpy as np
from scipy.special import gammaln

from ._utils import MaxIterWarning, as_matrix, as_vector, lam, logdet

__all__ = ['vb_logit_fit', 'vb_logit_fit_ard', 'vb_logit_fit_iter',
           'vb_logit_pred', 'vb_logit_pred_incr']


def vb_logit_fit(X, y, a0=1e-2, b0=1e-4):
    """Fit a logit model p(y = 1 | x, w) = 1 / (1 + exp(- w' x)) with a
    shrinkage prior on w.

    Parameters
    ----------
    X : (N, D) array
        Training input samples, one per row.
    y : (N,) array
        Corresponding output samples in {-1, 1}.
    a0, b0 : float, optional
        Shrinkage prior parameters. The defaults (a0 = 1e-2, b0 = 1e-4)
        result in a weak shrinkage prior.

    Returns
    -------
    w : (D,) array
        Posterior weight mean vector.
    V : (D, D) array
        Posterior weight covariance matrix.
    invV, logdetV : (D, D) array, float
        Inverse of V, and its log-determinant.
    E_a : float
        Mean E(a) of the shrinkage posterior.
    L : float
        Variational bound, lower-bounding the log-model evidence p(y | X).

    Notes
    -----
    The underlying generative model assumes a weight vector prior

        p(w | a) = N(w | 0, a^-1 I),

    and hyperprior

        p(a) = Gam(a | a0, b0).

    The function returns the parameters of the posterior

        p(w1 | X, y) = N(w1 | w, V).
    """
    X = as_matrix(X)
    y = as_vector(y)

    # pre-compute some constants
    N, D = X.shape
    max_iter = 500
    an = a0 + 0.5 * D
    gammaln_an_an = gammaln(an) + an
    t_w = 0.5 * (X.T @ y)

    # start first iteration kind of here, with xi = 0 -> lam_xi = 1/8
    lam_xi = np.ones(N) / 8
    E_a = a0 / b0
    invV = E_a * np.eye(D) + 2 * X.T @ (X * lam_xi[:, np.newaxis])
    V = np.linalg.inv(invV)
    w = V @ t_w
    bn = b0 + 0.5 * (w @ w + np.trace(V))
    L_last = (-N * np.log(2)
              + 0.5 * (w @ invV @ w - logdet(invV))
              - an / bn * b0 - an * np.log(bn) + gammaln_an_an)

    # update xi, bn, (V, w) iteratively
    for i in range(1, max_iter + 1):
        # update xi by EM-algorithm
        xi = np.sqrt(np.sum(X * (X @ (V + np.outer(w, w))), axis=1))
        lam_xi = lam(xi)

        # update posterior parameters of a based on xi
        bn = b0 + 0.5 * (w @ w + np.trace(V))
        E_a = an / bn

        # recompute posterior parameters of w
        invV = E_a * np.eye(D) + 2 * X.T @ (X * lam_xi[:, np.newaxis])
        V = np.linalg.inv(invV)
        logdetV = -logdet(invV)
        w = V @ t_w

        # variational bound, ignoring constant terms for now
        L = (-np.sum(np.log(1 + np.exp(-xi))) + np.sum(lam_xi * xi ** 2)
             + 0.5 * (w @ invV @ w + logdetV - np.sum(xi))
             - E_a * b0 - an * np.log(bn) + gammaln_an_an)

        # either stop if variational bound grows or change is < 0.001%
        # HACK ALARM: theoretically, the bound should never grow, and it doing
        # so points to numerical instabilities. As it seems, these start to
        # occur close to the optimal bound, which already points to a good
        # approximation.
        if (L_last > L) or (abs(L_last - L) < abs(0.00001 * L)):
            break
        L_last = L
    if i == max_iter:
        warnings.warn('Bayesian logistic regression reached maximum number '
                      'of iterations.', MaxIterWarning)

    # add constant terms to variational bound
    L = L - gammaln(a0) + a0 * np.log(b0)

    return w, V, invV, logdetV, E_a, L


def vb_logit_fit_ard(X, y, a0=1e-2, b0=1e-4):
    """Fit a logit model p(y = 1 | x, w) = 1 / (1 + exp(- w' x)) with
    automatic relevance determination (ARD) on w.

    Parameters
    ----------
    X : (N, D) array
        Training input samples, one per row.
    y : (N,) array
        Corresponding output samples in {-1, 1}.
    a0, b0 : float, optional
        Shrinkage prior parameters. The defaults (a0 = 1e-2, b0 = 1e-4)
        result in a weak shrinkage prior.

    Returns
    -------
    w : (D,) array
        Posterior weight mean vector.
    V : (D, D) array
        Posterior weight covariance matrix.
    invV, logdetV : (D, D) array, float
        Inverse of V, and its log-determinant.
    E_a : (D,) array
        Mean vector E(a) of the shrinkage posteriors.
    L : float
        Variational bound, lower-bounding the log-model evidence p(y | X).

    Notes
    -----
    The underlying generative model assumes a weight vector prior

        p(w_i | a_i) = N(w_i | 0, a_i^-1)

    and hyperpriors

        p(a_i) = Gam(a_i | a0, b0).

    The function returns the parameters of the posterior

        p(w1 | X, y) = N(w1 | w, V).
    """
    X = as_matrix(X)
    y = as_vector(y)

    # pre-compute some constants
    N, D = X.shape
    max_iter = 500
    an = a0 + 0.5
    D_gammaln_an_an = D * (gammaln(an) + an)
    t_w = 0.5 * (X.T @ y)

    # start first iteration kind of here, with xi = 0 -> lam_xi = 1/8
    lam_xi = np.ones(N) / 8
    E_a = np.ones(D) * a0 / b0
    invV = np.diag(E_a) + 2 * X.T @ (X * lam_xi[:, np.newaxis])
    V = np.linalg.inv(invV)
    w = V @ t_w
    bn = b0 + 0.5 * (w ** 2 + np.diag(V))
    L_last = (-N * np.log(2)
              + 0.5 * (w @ invV @ w - logdet(invV))
              - np.sum((b0 * an) / bn) - np.sum(an * np.log(bn))
              + D_gammaln_an_an)

    # update xi, bn, (V, w) iteratively
    for i in range(1, max_iter + 1):
        # update xi by EM-algorithm
        xi = np.sqrt(np.sum(X * (X @ (V + np.outer(w, w))), axis=1))
        lam_xi = lam(xi)

        # update posterior parameters of a based on xi
        bn = b0 + 0.5 * (w ** 2 + np.diag(V))
        E_a = an / bn

        # recompute posterior parameters of w
        invV = np.diag(E_a) + 2 * X.T @ (X * lam_xi[:, np.newaxis])
        V = np.linalg.inv(invV)
        logdetV = -logdet(invV)
        w = V @ t_w

        # variational bound, ignoring constant terms for now
        L = (-np.sum(np.log(1 + np.exp(-xi))) + np.sum(lam_xi * xi ** 2)
             + 0.5 * (w @ invV @ w + logdetV - np.sum(xi))
             - np.sum(b0 * E_a) - np.sum(an * np.log(bn)) + D_gammaln_an_an)

        # either stop if variational bound grows or change is < 0.001%
        # HACK ALARM: theoretically, the bound should never grow, and it doing
        # so points to numerical instabilities. As it seems, these start to
        # occur close to the optimal bound, which already points to a good
        # approximation.
        if (L_last > L) or (abs(L_last - L) < abs(0.00001 * L)):
            break
        L_last = L
    if i == max_iter:
        warnings.warn('Bayesian logistic regression reached maximum number '
                      'of iterations.', MaxIterWarning)

    # add constant terms to variational bound
    L = L - D * (gammaln(a0) - a0 * np.log(b0))

    return w, V, invV, logdetV, E_a, L


def vb_logit_fit_iter(X, y):
    """Fit a logit model p(y = 1 | x, w) = 1 / (1 + exp(- w' x)), processing
    the inputs one by one.

    Parameters
    ----------
    X : (N, D) array
        Training input samples, one per row.
    y : (N,) array
        Corresponding output samples in {-1, 1}.

    Returns
    -------
    w : (D,) array
        Posterior weight mean vector.
    V : (D, D) array
        Posterior weight covariance matrix.
    invV, logdetV : (D, D) array, float
        Inverse of V, and its log-determinant.

    Notes
    -----
    The underlying generative model assumes a weight vector prior

        p(w) = N(w | 0, D^-1 I),

    where D is the size of x. The function returns the parameters of the
    posterior

        p(w1 | X, y) = N(w1 | w, V).

    Compared to vb_logit_fit[_ard], this function does not use a hyperprior,
    iterates over the inputs separately rather than processing them all at
    once, and is therefore slower, but also computationally more stable as
    it avoids computing the inverse of possibly close-to-singular matrices.
    """
    X = as_matrix(X)
    y = as_vector(y)

    N, D = X.shape
    max_iter = 500

    # (more or less) uninformative prior
    V = np.eye(D) / D
    invV = np.eye(D) * D
    logdetV = -D * np.log(D)
    w = np.zeros(D)

    # iterate over all x separately
    for n in range(N):
        xn = X[n, :]

        # precompute values
        Vx = V @ xn
        VxVx = np.outer(Vx, Vx)
        c = xn @ Vx
        xx = np.outer(xn, xn)
        t_w = invV @ w + 0.5 * y[n] * xn

        # start iteration at xi = 0, lam_xi = 1/8
        V_xi = V - VxVx / (4 + c)
        invV_xi = invV + xx / 4
        logdetV_xi = logdetV - np.log(1 + c / 4)
        w = V_xi @ t_w

        L_last = 0.5 * (logdetV_xi + w @ invV_xi @ w) - np.log(2)

        for i in range(1, max_iter + 1):
            # update xi by EM algorithm
            xi = np.sqrt(xn @ (V_xi + np.outer(w, w)) @ xn)
            lam_xi = lam(xi)

            # Sherman-Morrison formula and Matrix determinant lemma
            V_xi = V - (2 * lam_xi / (1 + 2 * lam_xi * c)) * VxVx
            invV_xi = invV + 2 * lam_xi * xx
            logdetV_xi = logdetV - np.log(1 + 2 * lam_xi * c)
            w = V_xi @ t_w

            L = (0.5 * (logdetV_xi + w @ invV_xi @ w - xi)
                 - np.log(1 + np.exp(-xi)) + lam_xi * xi ** 2)

            # variational bound must grow!
            if L_last > L:
                print(f'Last bound {L_last:6.6f}, current bound {L:6.6f}')
                raise RuntimeError('Variational bound should not reduce')
            # stop if change in variation bound is < 0.001%
            if abs(L_last - L) < abs(0.00001 * L):
                break
            L_last = L
        if i == max_iter:
            warnings.warn('Bayesian logistic regression reached maximum '
                          'number of iterations.', MaxIterWarning)

        V = V_xi
        invV = invV_xi
        logdetV = logdetV_xi

    return w, V, invV, logdetV


def vb_logit_pred(X, w, V, invV):
    """Return p(y=1 | x, X, Y) for each row x of X, for a fitted Bayesian
    logit model.

    Parameters
    ----------
    X : (K, D) array
        Input samples, one per row.
    w : (D,) array
        Posterior weight mean.
    V : (D, D) array
        Posterior weight covariance matrix.
    invV : (D, D) array
        Inverse of V.
    w, V and invV are the fitted model parameters returned by
    vb_logit_fit[_*].

    Returns
    -------
    out : (K,) array
        p(y=1 | x, X, Y) for each row x of X.

    Notes
    -----
    The function assumes model parameters corresponding to the data
    likelihood

        p(y = 1 | x, w1) = 1 / (1 + exp(- w1' x)),

    with w, V, invV specifying the posterior parameters N(w1 | w, V).
    """
    X = as_matrix(X)
    w = as_vector(w)

    max_iter = 500
    N, Dx = X.shape

    # precompute some constants
    w_t = np.ones((N, 1)) * (w @ invV) + 0.5 * X  # w_t = V^-1 w + x / 2 as rows
    Vx = X @ V                                     # V x as rows
    VxxVwt = Vx * np.sum(w_t * Vx, axis=1)[:, np.newaxis]  # V x x^T V^T w_t
    Vwt = w_t @ V                                  # V w_t as rows
    xVx = np.sum(Vx * X, axis=1)                   # x^T V x as rows
    xVx2 = xVx ** 2                                # x^T V x x^T V x as rows

    # start first iteration with xi = 0, lam_xi = 1/8
    xi = np.zeros(N)
    lam_xi = lam(xi)
    a_xi = 1 / (4 + xVx)
    w_xi = Vwt - a_xi[:, np.newaxis] * VxxVwt
    logdetV_xi = -np.log(1 + xVx / 4)
    wVw_xi = (np.sum(w_xi * (w_xi @ invV), axis=1)
              + np.sum(w_xi * X, axis=1) ** 2 / 4)
    L_last = 0.5 * (np.sum(logdetV_xi) + np.sum(wVw_xi)) - N * np.log(2)

    # iterate to find xi's that maximise variational bound
    for i in range(1, max_iter + 1):
        # update xi by EM algorithm
        xi = np.sqrt(xVx - a_xi * xVx2 + np.sum(w_xi * X, axis=1) ** 2)
        lam_xi = lam(xi)

        # Sherman Morrison formula and Matrix determinant lemma
        a_xi = 2 * lam_xi / (1 + 2 * lam_xi * xVx)
        w_xi = Vwt - a_xi[:, np.newaxis] * VxxVwt
        logdetV_xi = -np.log(1 + 2 * lam_xi * xVx)

        # variational bound, omitting constant terms
        wVw_xi = (np.sum(w_xi * (w_xi @ invV), axis=1)
                  + 2 * lam_xi * (np.sum(w_xi * X, axis=1) ** 2))
        L = np.sum(0.5 * (logdetV_xi + wVw_xi - xi)
                   - np.log(1 + np.exp(-xi)) + lam_xi * xi ** 2)

        # variational bound must grow!
        if L_last > L:
            print(f'Last bound {L_last:6.6f}, current bound {L:6.6f}')
            raise RuntimeError('Variational bound should not reduce')
        # stop if change in variation bound is < 0.001%
        if abs(L_last - L) < abs(0.00001 * L):
            break
        L_last = L

    # posterior from optimal xi's
    out = (1 / (1 + np.exp(-xi)) / np.sqrt(1 + 2 * lam_xi * xVx)
           * np.exp(0.5 * (-xi - w @ invV @ w + wVw_xi) + lam_xi * xi ** 2))
    return out


def vb_logit_pred_incr(X, w, V, invV):
    """Return p(y=1 | x, X, Y) for each row x of X, for a fitted Bayesian
    logit model, iterating over the rows of X one by one.

    Parameters
    ----------
    X : (K, D) array
        Input samples, one per row.
    w : (D,) array
        Posterior weight mean.
    V : (D, D) array
        Posterior weight covariance matrix.
    invV : (D, D) array
        Inverse of V.
    w, V and invV are the fitted model parameters returned by
    vb_logit_fit[_*].

    Returns
    -------
    out : (K,) array
        p(y=1 | x, X, Y) for each row x of X.

    Notes
    -----
    The function assumes model parameters corresponding to the data
    likelihood

        p(y = 1 | x, w1) = 1 / (1 + exp(- w1' x)),

    with w, V, invV specifying the posterior parameters N(w1 | w, V). In
    contrast to vb_logit_pred, which computes the predictions for all rows
    of X simultaneously, this function iterates over the rows of X.
    """
    X = as_matrix(X)
    w = as_vector(w)

    max_iter = 500
    N = X.shape[0]
    out = np.zeros(N)

    # iterate over x, finding xi for each x separately
    for n in range(N):
        xn = X[n, :]

        # precompute values
        Vx = V @ xn
        VxVx = np.outer(Vx, Vx)
        c = xn @ Vx
        xx = np.outer(xn, xn)
        t_w = invV @ w + 0.5 * xn

        # start iteration at xi = 0, lam_xi = 1/8
        V_xi = V - VxVx / (4 + c)
        invV_xi = invV + xx / 4
        logdetV_xi = -np.log(1 + c / 4)
        w_xi = V_xi @ t_w
        L_last = 0.5 * (logdetV_xi + w_xi @ invV_xi @ w_xi) - np.log(2)

        # iterate to find xi that maximises variational bound
        for i in range(1, max_iter + 1):
            # update xi by EM algorithm
            xi = np.sqrt(xn @ (V_xi + np.outer(w_xi, w_xi)) @ xn)
            lam_xi = lam(xi)

            # Sherman-Morrison formula and Matrix determinant lemma
            V_xi = V - (2 * lam_xi / (1 + 2 * lam_xi * c)) * VxVx
            invV_xi = invV + 2 * lam_xi * xx
            logdetV_xi = -np.log(1 + 2 * lam_xi * c)
            w_xi = V_xi @ t_w

            # variational bound, omitting constant terms
            L = (0.5 * (logdetV_xi + w_xi @ invV_xi @ w_xi - xi)
                 - np.log(1 + np.exp(-xi)) + lam_xi * xi ** 2)

            # variational bound must grow!
            if L_last > L:
                print(f'Last bound {L_last:6.6f}, current bound {L:6.6f}')
                raise RuntimeError('Variational bound should not reduce')
            # stop if change in variation bound is < 0.001%
            if abs(L_last - L) < abs(0.00001 * L):
                break
            L_last = L

        # p(y=1 | x, X, Y), using again Matrix determinant lemma
        out[n] = (1 / (1 + np.exp(-xi)) / np.sqrt(1 + 2 * lam_xi * c)
                  * np.exp(-0.5 * xi + lam_xi * xi ** 2
                           - 0.5 * w @ invV @ w
                           + 0.5 * w_xi @ invV_xi @ w_xi))
    return out
