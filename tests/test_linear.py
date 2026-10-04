"""Unit tests for vb_linear_fit, vb_linear_fit_ard and vb_linear_pred.

Port of test/vb_linear_*_test.m.
"""

import numpy as np
import pytest

from vblinlogit import vb_linear_fit, vb_linear_fit_ard, vb_linear_pred

# settings
WGEN = np.array([1.0, 2.0, 3.0])  # test weight vector
TAU = 2                           # test noise precision
N_SMALL = 50                      # training set (small)
N_LARGE = 10000                   # training set (large)
A0, B0, C0, D0 = 1e-2, 1e-4, 1e-2, 1e-4

FITS = [vb_linear_fit, vb_linear_fit_ard]


def gen_X(N):
    return np.linspace(0, 1, N)[:, None] ** np.arange(len(WGEN))


def gen_y(X, rng):
    return X @ WGEN + np.sqrt(1 / TAU) * rng.standard_normal(X.shape[0])


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.mark.parametrize('fit', FITS)
def test_fit_return_sizes(fit, rng):
    X = gen_X(N_SMALL)
    w, V, invV, logdetV, an, bn, E_a, L = fit(X, gen_y(X, rng))
    D = len(WGEN)
    assert w.shape == WGEN.shape
    assert V.shape == (D, D)
    assert invV.shape == (D, D)
    assert np.ndim(logdetV) == 0
    assert np.ndim(an) == 0
    assert np.ndim(bn) == 0
    if fit is vb_linear_fit:
        assert np.ndim(E_a) == 0
    else:
        assert E_a.shape == WGEN.shape
    assert np.ndim(L) == 0


@pytest.mark.parametrize('fit', FITS)
def test_fit_consistency(fit, rng):
    X = gen_X(N_SMALL)
    y = gen_y(X, rng)
    res1 = fit(X, y)
    res2 = fit(X, y)
    for name, a, b in zip(['w', 'V', 'invV', 'logdetV', 'an', 'bn', 'E_a',
                           'L'], res1, res2):
        assert np.linalg.norm(np.atleast_1d(a - b)) <= 1e-10, name


@pytest.mark.parametrize('fit', FITS)
def test_fit_optional_arguments(fit, rng):
    X = gen_X(N_SMALL)
    y = gen_y(X, rng)
    w1, V1 = fit(X, y)[:2]
    for args in [(A0,), (A0, B0), (A0, B0, C0), (A0, B0, C0, D0)]:
        w2, V2 = fit(X, y, *args)[:2]
        assert np.linalg.norm(w1 - w2) <= 1e-10, args
        assert np.linalg.norm(V1 - V2) <= 1e-10, args


@pytest.mark.parametrize('fit', FITS)
def test_fit_weight_estimates(fit, rng):
    X = gen_X(N_LARGE)
    w = fit(X, gen_y(X, rng))[0]
    assert np.linalg.norm(w - WGEN) <= 0.5


def test_fit_accepts_column_vector_y(rng):
    X = gen_X(N_SMALL)
    y = gen_y(X, rng)
    res1 = vb_linear_fit_ard(X, y)
    res2 = vb_linear_fit_ard(X, y[:, None])
    np.testing.assert_allclose(res1[0], res2[0])


def test_pred_return_sizes(rng):
    X = gen_X(N_SMALL)
    w, V, _, _, an, bn = vb_linear_fit_ard(X, gen_y(X, rng))[:6]
    mu, lam, nu = vb_linear_pred(X, w, V, an, bn)
    assert mu.shape == (N_SMALL,)
    assert lam.shape == (N_SMALL,)
    assert np.ndim(nu) == 0


def test_pred_consistency(rng):
    X = gen_X(N_SMALL)
    w, V, _, _, an, bn = vb_linear_fit_ard(X, gen_y(X, rng))[:6]
    mu1, lam1, nu1 = vb_linear_pred(X, w, V, an, bn)
    mu2, lam2, nu2 = vb_linear_pred(X, w, V, an, bn)
    assert np.linalg.norm(mu1 - mu2) <= 1e-10
    assert np.linalg.norm(lam1 - lam2) <= 1e-10
    assert abs(nu1 - nu2) <= 1e-10


def test_pred_output_predictions(rng):
    X = gen_X(N_LARGE)
    y = gen_y(X, rng)
    w, V, _, _, an, bn = vb_linear_fit_ard(X, y)[:6]
    mu = vb_linear_pred(X, w, V, an, bn)[0]
    assert np.mean((mu - y) ** 2) <= 1
