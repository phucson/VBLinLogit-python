"""Smoke tests: every example script runs to completion without figures.

The high-dimensional examples take a while; skip them with
``pytest -m "not slow"``.
"""

import importlib
import os
import sys

import matplotlib
import pytest

matplotlib.use('Agg')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'examples'))

FAST = ['vb_linear_example', 'vb_linear_example_highdim',
        'vb_linear_example_modelsel', 'vb_logit_example',
        'vb_logit_example_coeff', 'vb_logit_example_modelsel']
SLOW = ['vb_linear_example_sparse', 'vb_logit_example_highdim']


@pytest.mark.parametrize('name', FAST + [pytest.param(n, marks=pytest.mark.slow)
                                         for n in SLOW])
def test_example_runs(name, tmp_path):
    module = importlib.import_module(name)
    result = module.main(save_dir=str(tmp_path), show=False)
    assert result
    assert any(f.suffix == '.png' for f in tmp_path.iterdir())


def test_modelsel_examples_recover_true_order():
    import vb_linear_example_modelsel
    import vb_logit_example_modelsel
    assert vb_linear_example_modelsel.main(show=False)['D_best'] == 3
    assert vb_logit_example_modelsel.main(show=False)['D_best'] == 3
