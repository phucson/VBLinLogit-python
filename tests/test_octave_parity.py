"""Check that the Python port reproduces the original MATLAB/Octave outputs.

The reference outputs in ``data/octave_reference.npz`` were produced by
running the original MATLAB code (``reference/matlab/src``) under GNU Octave
on fixed datasets; see ``tools/generate_octave_reference.py``.
"""

import os

import numpy as np
import pytest

from vblinlogit import (logdet, vb_linear_fit, vb_linear_fit_ard,
                        vb_linear_pred, vb_logit_fit, vb_logit_fit_ard,
                        vb_logit_fit_iter, vb_logit_pred, vb_logit_pred_incr)

REF_FILE = os.path.join(os.path.dirname(__file__), 'data',
                        'octave_reference.npz')
REF = np.load(REF_FILE)

RTOL = 1e-8
ATOL = 1e-10

LINEAR_NAMES = ['w', 'V', 'invV', 'logdetV', 'an', 'bn', 'E_a', 'L']
LOGIT_NAMES = ['w', 'V', 'invV', 'logdetV', 'E_a', 'L']
ITER_NAMES = ['w', 'V', 'invV', 'logdetV']


def inp(name):
    return REF['in_' + name]


def assert_matches(prefix, names, result):
    assert len(result) == len(names)
    for name, value in zip(names, result):
        expected = np.squeeze(REF[f'out_{prefix}_{name}'])
        np.testing.assert_allclose(np.squeeze(value), expected,
                                   rtol=RTOL, atol=ATOL,
                                   err_msg=f'{prefix}: {name}')


@pytest.mark.parametrize('prefix, func, data, args', [
    ('lf1', vb_linear_fit, 'lin1', ()),
    ('lf2', vb_linear_fit, 'lin2', ()),
    ('lf3', vb_linear_fit, 'lin2', (1, 2, 0.5, 0.1)),
    ('la1', vb_linear_fit_ard, 'lin1', ()),
    ('la2', vb_linear_fit_ard, 'lin2', ()),
    ('la3', vb_linear_fit_ard, 'lin2', (1, 2, 0.5, 0.1)),
])
def test_linear_fit(prefix, func, data, args):
    res = func(inp(data + '_X'), inp(data + '_y'), *args)
    assert_matches(prefix, LINEAR_NAMES, res)


@pytest.mark.parametrize('prefix, fit', [('lp1', 'la2'), ('lp2', 'lf2')])
def test_linear_pred(prefix, fit):
    w, V, an, bn = (REF[f'out_{fit}_{n}'] for n in ['w', 'V', 'an', 'bn'])
    res = vb_linear_pred(inp('lin_pred_X'), w, V, an.item(), bn.item())
    assert_matches(prefix, ['mu', 'lambda', 'nu'], res)


@pytest.mark.parametrize('prefix, func, data, args', [
    ('gf1', vb_logit_fit, 'log1', ()),
    ('gf2', vb_logit_fit, 'log2', ()),
    ('gf3', vb_logit_fit, 'log2', (1, 0.5)),
    ('ga1', vb_logit_fit_ard, 'log1', ()),
    ('ga2', vb_logit_fit_ard, 'log2', ()),
    ('ga3', vb_logit_fit_ard, 'log2', (1, 0.5)),
])
def test_logit_fit(prefix, func, data, args):
    res = func(inp(data + '_X'), inp(data + '_y'), *args)
    assert_matches(prefix, LOGIT_NAMES, res)


@pytest.mark.parametrize('prefix, data', [('gi1', 'log1'), ('gi2', 'log2')])
def test_logit_fit_iter(prefix, data):
    res = vb_logit_fit_iter(inp(data + '_X'), inp(data + '_y'))
    assert_matches(prefix, ITER_NAMES, res)


@pytest.mark.parametrize('func, prefixes', [
    (vb_logit_pred, ['gp1', 'gp2', 'gp3']),
    (vb_logit_pred_incr, ['gn1', 'gn2', 'gn3']),
])
def test_logit_pred(func, prefixes):
    cases = [('log_pred_X', 'gf2'), ('log_pred_X', 'ga2'), ('log1_X', 'gi1')]
    for prefix, (X, fit) in zip(prefixes, cases):
        w, V, invV = (REF[f'out_{fit}_{n}'] for n in ['w', 'V', 'invV'])
        out = func(inp(X), w, V, invV)
        np.testing.assert_allclose(out, np.squeeze(REF['out_' + prefix]),
                                   rtol=RTOL, atol=ATOL, err_msg=prefix)


def test_logdet():
    X = inp('lin2_X')
    np.testing.assert_allclose(logdet(X.T @ X), REF['out_logdet1'].item(),
                               rtol=1e-12)
