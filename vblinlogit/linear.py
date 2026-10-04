"""Variational Bayesian linear regression.

Python port of ``vb_linear_fit.m``, ``vb_linear_fit_ard.m`` and
``vb_linear_pred.m``.

Copyright (c) 2013-2019, Jan Drugowitsch (original MATLAB code)
All rights reserved.
See the file LICENSE for licensing information.
"""

import warnings

import numpy as np
from scipy.special import gammaln

from ._utils import MaxIterWarning, as_matrix, as_vector, logdet

__all__ = ['vb_linear_fit', 'vb_linear_fit_ard', 'vb_linear_pred']


def vb_linear_fit(X, y, a0=1e-2, b0=1e-4, c0=1e-2, d0=1e-4):
    """Estimate w such that y = Xw, using Bayesian regularisation.

    Parameters
    ----------
    X : (N, D) array
        Training input samples, one per row.
    y : (N,) array
        Corresponding output samples.
    a0, b0 : float, optional
        Prior parameters of the noise precision.
    c0, d0 : float, optional
        Hyper-prior shrinkage parameters.
    The defaults (a0 = 1e-2, b0 = 1e-4, c0 = 1e-2, d0 = 1e-4) result in an
    uninformative prior.

    Returns
    -------
    w : (D,) array
        Posterior weight mean vector.
    V : (D, D) array
        Posterior weight covariance matrix.
    invV, logdetV : (D, D) array, float
        Inverse of V, and its log-determinant.
    an, bn : float
        Posterior parameters of the noise precision.
    E_a : float
        Mean E(alpha) of the shrinkage hyper-posterior.
    L : float
        Variational bound, lower-bounding the log-model evidence p(y | X).

    Notes
    -----
    The underlying generative model assumes

        p(y | x, w, tau) = N(y | w'x, tau^-1),

    with x and y being the rows of the given X and y. w and tau are assigned
    the conjugate normal inverse-gamma prior

        p(w, tau | alpha) = N(w | 0, (tau alpha)^-1 I) Gam(tau | a0, b0),

    with the hyper-prior

        p(alpha) = p(alpha | c0, d0).

    The returned posterior parameters (computed by variational Bayesian
    inference) determine a posterior of the form

        N(w1 | w, tau^-1 V) Gam(tau | an, bn).
    """
    X = as_matrix(X)
    y = as_vector(y)

    # pre-process data
    N, D = X.shape
    X_corr = X.T @ X
    Xy_corr = X.T @ y
    an = a0 + N / 2
    gammaln_an = gammaln(an)
    cn = c0 + D / 2
    gammaln_cn = gammaln(cn)

    # iterate to find hyperparameters
    L_last = -np.finfo(float).max
    max_iter = 500
    E_a = c0 / d0
    for it in range(1, max_iter + 1):
        # covariance and weight of linear model
        invV = E_a * np.eye(D) + X_corr
        V = np.linalg.inv(invV)
        logdetV = -logdet(invV)
        w = V @ Xy_corr
        # parameters of noise model (an remains constant)
        sse = np.sum((X @ w - y) ** 2)
        bn = b0 + 0.5 * (sse + E_a * (w @ w))
        E_t = an / bn

        # hyperparameters of covariance prior (cn remains constant)
        dn = d0 + 0.5 * (E_t * (w @ w) + np.trace(V))
        E_a = cn / dn

        # variational bound, ignoring constant terms for now
        L = (-0.5 * (E_t * sse + np.sum(X * (X @ V))) + 0.5 * logdetV
             - b0 * E_t + gammaln_an - an * np.log(bn) + an
             + gammaln_cn - cn * np.log(dn))

        # variational bound must grow!
        if L_last > L:
            print(f'Last bound {L_last:6.6f}, current bound {L:6.6f}')
            raise RuntimeError('Variational bound should not reduce')
        # stop if change in variation bound is < 0.001%
        if abs(L_last - L) < abs(0.00001 * L):
            break
        L_last = L
    if it == max_iter:
        warnings.warn('Bayesian linear regression reached maximum number '
                      'of iterations.', MaxIterWarning)

    # augment variational bound with constant terms
    L = (L - 0.5 * (N * np.log(2 * np.pi) - D) - gammaln(a0) + a0 * np.log(b0)
         - gammaln(c0) + c0 * np.log(d0))

    return w, V, invV, logdetV, an, bn, E_a, L


def vb_linear_fit_ard(X, y, a0=1e-2, b0=1e-4, c0=1e-2, d0=1e-4):
    """Estimate w such that y = Xw, using Bayesian regularisation with
    automatic relevance determination (ARD).

    Parameters
    ----------
    X : (N, D) array
        Training input samples, one per row.
    y : (N,) array
        Corresponding output samples.
    a0, b0 : float, optional
        Prior parameters of the noise precision.
    c0, d0 : float, optional
        Hyper-prior shrinkage parameters.
    The defaults (a0 = 1e-2, b0 = 1e-4, c0 = 1e-2, d0 = 1e-4) result in an
    uninformative prior.

    Returns
    -------
    w : (D,) array
        Posterior weight mean vector.
    V : (D, D) array
        Posterior weight covariance matrix.
    invV, logdetV : (D, D) array, float
        Inverse of V, and its log-determinant.
    an, bn : float
        Posterior parameters of the noise precision.
    E_a : (D,) array
        Mean vector E(alpha) of the shrinkage hyper-posterior.
    L : float
        Variational bound, lower-bounding the log-model evidence p(y | X).

    Notes
    -----
    The underlying generative model assumes

        p(y | x, w, tau) = N(y | w'x, tau^-1),

    with x and y being the rows of the given X and y. w and tau are assigned
    the conjugate normal inverse-gamma prior

        p(w, tau | alpha) = N(w | 0, (tau A)^-1) Gam(tau | a0, b0),

    with A being a diagonal matrix with the vector
    alpha = (alpha_1, ..., alpha_D)' along its diagonal, and the hyper-priors

        p(alpha_i) = p(alpha | c0, d0).

    The returned posterior parameters (computed by variational Bayesian
    inference) determine a posterior of the form

        N(w1 | w, tau^-1 V) Gam(tau | an, bn).
    """
    X = as_matrix(X)
    y = as_vector(y)

    # pre-process data
    N, D = X.shape
    X_corr = X.T @ X
    Xy_corr = X.T @ y
    an = a0 + N / 2
    gammaln_an = gammaln(an)
    cn = c0 + 1 / 2
    D_gammaln_cn = D * gammaln(cn)

    # iterate to find hyperparameters
    L_last = -np.finfo(float).max
    max_iter = 500
    E_a = np.ones(D) * c0 / d0
    for it in range(1, max_iter + 1):
        # covariance and weight of linear model
        invV = np.diag(E_a) + X_corr
        V = np.linalg.inv(invV)
        logdetV = -logdet(invV)
        w = V @ Xy_corr
        # parameters of noise model (an remains constant)
        sse = np.sum((X @ w - y) ** 2)
        bn = b0 + 0.5 * (sse + np.sum(w ** 2 * E_a))
        E_t = an / bn

        # hyperparameters of covariance prior (cn remains constant)
        dn = d0 + 0.5 * (E_t * w ** 2 + np.diag(V))
        E_a = cn / dn

        # variational bound, ignoring constant terms for now
        L = (-0.5 * (E_t * sse + np.sum(X * (X @ V))) + 0.5 * logdetV
             - b0 * E_t + gammaln_an - an * np.log(bn) + an
             + D_gammaln_cn - cn * np.sum(np.log(dn)))

        # variational bound must grow!
        if L_last > L:
            print(f'Last bound {L_last:6.6f}, current bound {L:6.6f}')
            raise RuntimeError('Variational bound should not reduce')
        # stop if change in variation bound is < 0.001%
        if abs(L_last - L) < abs(0.00001 * L):
            break
        L_last = L
    if it == max_iter:
        warnings.warn('Bayesian linear regression reached maximum number '
                      'of iterations.', MaxIterWarning)

    # augment variational bound with constant terms
    L = (L - 0.5 * (N * np.log(2 * np.pi) - D) - gammaln(a0) + a0 * np.log(b0)
         + D * (-gammaln(c0) + c0 * np.log(d0)))

    return w, V, invV, logdetV, an, bn, E_a, L


def vb_linear_pred(X, w, V, an, bn):
    """Posterior predictive for vb_linear_fit[_ard], for inputs as rows of X.

    Parameters
    ----------
    X : (K, D) array
        Input samples, one per row.
    w : (D,) array
        Posterior weight mean.
    V : (D, D) array
        Posterior weight covariance matrix.
    an, bn : float
        Posterior parameters of the noise precision.
    w, V, an and bn are the fitted model parameters returned by
    vb_linear_fit[_ard].

    Returns
    -------
    mu : (K,) array
        Predicted output means.
    lam : (K,) array
        Predicted output precisions.
    nu : float
        Predicted output degrees of freedom.

    Notes
    -----
    The predictive posteriors are of the form

        St(y | mu, lambda, nu),

    which is a Student's t distribution with mean mu, precision lambda, and
    nu degrees of freedom. mu and lambda are vectors, one element per input
    x. nu is a scalar as it is the same for all x.
    """
    X = as_matrix(X)
    w = as_vector(w)
    mu = X @ w
    lam = (an / bn) / (1 + np.sum(X * (X @ V), axis=1))
    nu = 2 * an
    return mu, lam, nu
