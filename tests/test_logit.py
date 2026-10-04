"""Unit tests for vb_logit_fit, vb_logit_fit_ard, vb_logit_fit_iter,
vb_logit_pred and vb_logit_pred_incr.

Port of test/vb_logit_*_test.m.
"""

import numpy as np
import pytest

from vblinlogit import (vb_logit_fit, vb_logit_fit_ard, vb_logit_fit_iter,
                        vb_logit_pred, vb_logit_pred_incr)
from vblinlogit._utils import lam

# settings
WGEN = np.array([1.0, 2.0, 3.0])  # test weight vector
N_SMALL = 50                      # training set (small)
N_LARGE = 100000                  # training set (large)
N_LARGE_INCR = 10000              # training set (large, vb_logit_pred_incr)
A0, B0 = 1e-2, 1e-4


def gen_X(N):
    return np.linspace(0, 1, N)[:, None] ** np.arange(len(WGEN))


def gen_y(X, rng):
    return 2.0 * (rng.random(X.shape[0]) < 1 / (1 + np.exp(-X @ WGEN))) - 1


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# --- model fitting ---

@pytest.mark.parametrize('fit', [vb_logit_fit, vb_logit_fit_ard])
def test_fit_return_sizes(fit, rng):
    X = gen_X(N_SMALL)
    w, V, invV, logdetV, E_a, L = fit(X, gen_y(X, rng))
    D = len(WGEN)
    assert w.shape == WGEN.shape
    assert V.shape == (D, D)
    assert invV.shape == (D, D)
    assert np.ndim(logdetV) == 0
    if fit is vb_logit_fit:
        assert np.ndim(E_a) == 0
    else:
        assert E_a.shape == WGEN.shape
    assert np.ndim(L) == 0


def test_fit_iter_return_sizes(rng):
    X = gen_X(N_SMALL)
    w, V, invV, logdetV = vb_logit_fit_iter(X, gen_y(X, rng))
    D = len(WGEN)
    assert w.shape == WGEN.shape
    assert V.shape == (D, D)
    assert invV.shape == (D, D)
    assert np.ndim(logdetV) == 0


@pytest.mark.parametrize('fit', [vb_logit_fit, vb_logit_fit_ard,
                                 vb_logit_fit_iter])
def test_fit_consistency(fit, rng):
    X = gen_X(N_SMALL)
    y = gen_y(X, rng)
    res1 = fit(X, y)
    res2 = fit(X, y)
    for name, a, b in zip(['w', 'V', 'invV', 'logdetV', 'E_a', 'L'],
                          res1, res2):
        assert np.linalg.norm(np.atleast_1d(a - b)) <= 1e-10, name


@pytest.mark.parametrize('fit', [vb_logit_fit, vb_logit_fit_ard])
def test_fit_optional_arguments(fit, rng):
    X = gen_X(N_SMALL)
    y = gen_y(X, rng)
    w1, V1 = fit(X, y)[:2]
    for args in [(A0,), (A0, B0)]:
        w2, V2 = fit(X, y, *args)[:2]
        assert np.linalg.norm(w1 - w2) <= 1e-10, args
        assert np.linalg.norm(V1 - V2) <= 1e-10, args


@pytest.mark.parametrize('fit', [vb_logit_fit, vb_logit_fit_ard,
                                 vb_logit_fit_iter])
def test_fit_weight_estimates(fit, rng):
    X = gen_X(N_LARGE)
    w = fit(X, gen_y(X, rng))[0]
    assert np.linalg.norm(w - WGEN) <= 1


# --- predictions ---

@pytest.mark.parametrize('pred', [vb_logit_pred, vb_logit_pred_incr])
def test_pred_return_sizes(pred, rng):
    X = gen_X(N_SMALL)
    w, V, invV = vb_logit_fit(X, gen_y(X, rng))[:3]
    py = pred(X, w, V, invV)
    assert py.shape == (N_SMALL,)


@pytest.mark.parametrize('pred', [vb_logit_pred, vb_logit_pred_incr])
def test_pred_consistency(pred, rng):
    X = gen_X(N_SMALL)
    w, V, invV = vb_logit_fit(X, gen_y(X, rng))[:3]
    py1 = pred(X, w, V, invV)
    py2 = pred(X, w, V, invV)
    assert np.linalg.norm(py1 - py2) <= 1e-10


@pytest.mark.parametrize('pred, N', [(vb_logit_pred, N_LARGE),
                                     (vb_logit_pred_incr, N_LARGE_INCR)])
def test_pred_output_predictions(pred, N, rng):
    X = gen_X(N)
    y = gen_y(X, rng)
    w, V, invV = vb_logit_fit(X, y)[:3]
    ypred = 2 * (pred(X, w, V, invV) > 0.5) - 1
    assert np.mean(np.abs(ypred - y)) <= 0.3


def test_pred_and_pred_incr_agree(rng):
    # both compute the same quantity, once vectorised, once per input
    X = gen_X(200)
    w, V, invV = vb_logit_fit(X, gen_y(X, rng))[:3]
    np.testing.assert_allclose(vb_logit_pred(X, w, V, invV),
                               vb_logit_pred_incr(X, w, V, invV), rtol=1e-6)


def test_lam():
    assert lam(0.0) == 1 / 8
    np.testing.assert_allclose(lam(np.array([0.0, 2.0])),
                               [1 / 8, np.tanh(1) / 8])
