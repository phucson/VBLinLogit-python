import numpy as np


def vb_logit_pred(X, w, V, invV):
    """out = vb_logit_pred(X, w, V, invV)

    returns a vector containing p(y=1 | x, X, Y) for x = each row in the
    given X, for a fitted Bayesian logit model.

    The function expects the arguments
    - X: K x D matrix of K input samples, one per row
    - w: D-element posterior weight mean
    - V: D x D posterior weight covariance matrix
    - invV: inverse of V
    w, V and invV are the fitted model parameters returned by
    vb_logit_fit[_*].

    It returns
    - out: K-element vector, with p(y=1 | x, X, Y) as each element.

    The function assumes model parameters corresponding to the data
    likelihood

    p(y = 1 | x, w1) = 1 / (1 + exp(- w1' * x)),

    with w, V, invV specifying the posterior parameters N(w1 | w, V).

    Copyright (c) 2013-2019, Jan Drugowitsch
    All rights reserved.
    See the file LICENSE for licensing information.
    """

    max_iter = 500
    N, Dx = X.shape

    ## precompute some constants
    w_t = np.ones((N, 1)) * (w.T @ invV) + 0.5 * X  # w_t = V^-1 w + x / 2 as rows
    Vx = X @ V                                      # W x as rows
    VxxVwt = Vx * (np.sum(w_t * Vx, 1)[:, None] * np.ones((1, Dx)))  # V x x^T V^T w_t as rows
    Vwt = w_t @ V                                   # V w_t as rows
    xVx = np.sum(Vx * X, 1)                         # x^T V x as rows
    xVx2 = xVx ** 2                                 # x^T V x x^T V x as rows

    ## start first iteration with xi = 0, lam_xi = 1/8
    xi = np.zeros(N)
    lam_xi = lam(xi)
    a_xi = 1 / (4 + xVx)
    w_xi = Vwt - (a_xi[:, None] * np.ones((1, Dx))) * VxxVwt
    logdetV_xi = - np.log(1 + xVx / 4)
    wVw_xi = np.sum(w_xi * (w_xi @ invV), 1) + np.sum(w_xi * X, 1) ** 2 / 4
    L_last = 0.5 * (np.sum(logdetV_xi) + np.sum(wVw_xi)) - N * np.log(2)


    ## iterate to from xi's that maximise variational bound
    for i in range(1, max_iter + 1):
        # update xi by EM algorithm
        xi = np.sqrt(xVx - a_xi * xVx2 + np.sum(w_xi * X, 1) ** 2)
        lam_xi = lam(xi)

        # Sherman Morrison formula and Matrix determinant lemma
        a_xi = 2 * lam_xi / (1 + 2 * lam_xi * xVx)
        w_xi = Vwt - (a_xi[:, None] * np.ones((1, Dx))) * VxxVwt
        logdetV_xi = - np.log(1 + 2 * lam_xi * xVx)

        # variational bound, omitting constant terms
        wVw_xi = np.sum(w_xi * (w_xi @ invV), 1) \
            + 2 * lam_xi * (np.sum(w_xi * X, 1) ** 2)
        L = np.sum(0.5 * (logdetV_xi + wVw_xi - xi)
                   - np.log(1 + np.exp(-xi)) + lam_xi * xi ** 2)

        # variational bound must grow!
        if L_last > L:
            print('Last bound %6.6f, current bound %6.6f' % (L_last, L))
            raise RuntimeError('Variational bound should not reduce')
        # stop if change in variation bound is < 0.001%
        if abs(L_last - L) < abs(0.00001 * L):
            break
        L_last = L


    ## posterior from optimal xi's
    out = 1 / (1 + np.exp(-xi)) / np.sqrt(1 + 2 * lam_xi * xVx) \
        * np.exp(0.5 * (-xi - w.T @ invV @ w + wVw_xi) + lam_xi * xi ** 2)

    return out


def lam(xi):
    # returns 1 / (4 * xi) * tanh(xi / 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.tanh(xi / 2) / (4 * xi)
    # fix values where xi = 0
    out[np.isnan(out)] = 1/8
    return out
