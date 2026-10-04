"""Generate reference outputs of the original MATLAB code using GNU Octave.

The script creates a set of fixed datasets, runs the original MATLAB
functions in ``reference/matlab/src`` on them under Octave, and stores
inputs and outputs in ``tests/data/octave_reference.npz``. The test
``tests/test_octave_parity.py`` then checks that the Python port reproduces
these outputs.

Usage:
    python tools/generate_octave_reference.py [--octave /path/to/octave-cli]
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from scipy.io import loadmat, savemat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATLAB_SRC = os.path.join(ROOT, 'reference', 'matlab', 'src')
OUT_FILE = os.path.join(ROOT, 'tests', 'data', 'octave_reference.npz')


def make_datasets():
    """Return a dict of named datasets (all arrays are 2D, as for MATLAB)."""
    rng = np.random.default_rng(20131022)
    data = {}

    # linear regression: polynomial basis, as in the unit tests
    wgen = np.array([1.0, 2.0, 3.0])
    x = np.linspace(0, 1, 50)
    X = x[:, None] ** np.arange(3)
    data['lin1_X'] = X
    data['lin1_y'] = (X @ wgen + np.sqrt(0.5) * rng.standard_normal(50))[:, None]
    # linear regression: higher-dimensional, sparse weights
    N, D = 80, 25
    w = np.concatenate([rng.standard_normal(5), np.zeros(D - 5)])
    X = rng.random((N, D)) - 0.5
    data['lin2_X'] = X
    data['lin2_y'] = (X @ w + rng.standard_normal(N))[:, None]
    data['lin_pred_X'] = rng.random((30, D)) - 0.5

    # logistic regression: polynomial basis, as in the unit tests
    x = np.linspace(0, 1, 200)
    X = x[:, None] ** np.arange(3)
    py = 1 / (1 + np.exp(-X @ wgen))
    data['log1_X'] = X
    data['log1_y'] = (2.0 * (rng.random(200) < py) - 1)[:, None]
    # logistic regression: higher-dimensional, sparse weights
    N, D = 300, 15
    w = np.concatenate([2 * rng.standard_normal(4), np.zeros(D - 4)])
    X = np.hstack([np.ones((N, 1)), rng.standard_normal((N, D - 1))])
    py = 1 / (1 + np.exp(-X @ w))
    data['log2_X'] = X
    data['log2_y'] = (2.0 * (rng.random(N) < py) - 1)[:, None]
    data['log_pred_X'] = np.hstack([np.ones((40, 1)),
                                    rng.standard_normal((40, D - 1))])
    return data


OCTAVE_SCRIPT = r"""
addpath('{src}');
load('{inp}');
r = struct();
% --- linear regression ---
[r.lf1_w, r.lf1_V, r.lf1_invV, r.lf1_logdetV, r.lf1_an, r.lf1_bn, r.lf1_E_a, r.lf1_L] = ...
    vb_linear_fit(lin1_X, lin1_y);
[r.lf2_w, r.lf2_V, r.lf2_invV, r.lf2_logdetV, r.lf2_an, r.lf2_bn, r.lf2_E_a, r.lf2_L] = ...
    vb_linear_fit(lin2_X, lin2_y);
[r.lf3_w, r.lf3_V, r.lf3_invV, r.lf3_logdetV, r.lf3_an, r.lf3_bn, r.lf3_E_a, r.lf3_L] = ...
    vb_linear_fit(lin2_X, lin2_y, 1, 2, 0.5, 0.1);
[r.la1_w, r.la1_V, r.la1_invV, r.la1_logdetV, r.la1_an, r.la1_bn, r.la1_E_a, r.la1_L] = ...
    vb_linear_fit_ard(lin1_X, lin1_y);
[r.la2_w, r.la2_V, r.la2_invV, r.la2_logdetV, r.la2_an, r.la2_bn, r.la2_E_a, r.la2_L] = ...
    vb_linear_fit_ard(lin2_X, lin2_y);
[r.la3_w, r.la3_V, r.la3_invV, r.la3_logdetV, r.la3_an, r.la3_bn, r.la3_E_a, r.la3_L] = ...
    vb_linear_fit_ard(lin2_X, lin2_y, 1, 2, 0.5, 0.1);
[r.lp1_mu, r.lp1_lambda, r.lp1_nu] = ...
    vb_linear_pred(lin_pred_X, r.la2_w, r.la2_V, r.la2_an, r.la2_bn);
[r.lp2_mu, r.lp2_lambda, r.lp2_nu] = ...
    vb_linear_pred(lin_pred_X, r.lf2_w, r.lf2_V, r.lf2_an, r.lf2_bn);
% --- logistic regression ---
[r.gf1_w, r.gf1_V, r.gf1_invV, r.gf1_logdetV, r.gf1_E_a, r.gf1_L] = vb_logit_fit(log1_X, log1_y);
[r.gf2_w, r.gf2_V, r.gf2_invV, r.gf2_logdetV, r.gf2_E_a, r.gf2_L] = vb_logit_fit(log2_X, log2_y);
[r.gf3_w, r.gf3_V, r.gf3_invV, r.gf3_logdetV, r.gf3_E_a, r.gf3_L] = vb_logit_fit(log2_X, log2_y, 1, 0.5);
[r.ga1_w, r.ga1_V, r.ga1_invV, r.ga1_logdetV, r.ga1_E_a, r.ga1_L] = vb_logit_fit_ard(log1_X, log1_y);
[r.ga2_w, r.ga2_V, r.ga2_invV, r.ga2_logdetV, r.ga2_E_a, r.ga2_L] = vb_logit_fit_ard(log2_X, log2_y);
[r.ga3_w, r.ga3_V, r.ga3_invV, r.ga3_logdetV, r.ga3_E_a, r.ga3_L] = vb_logit_fit_ard(log2_X, log2_y, 1, 0.5);
[r.gi1_w, r.gi1_V, r.gi1_invV, r.gi1_logdetV] = vb_logit_fit_iter(log1_X, log1_y);
[r.gi2_w, r.gi2_V, r.gi2_invV, r.gi2_logdetV] = vb_logit_fit_iter(log2_X, log2_y);
r.gp1 = vb_logit_pred(log_pred_X, r.gf2_w, r.gf2_V, r.gf2_invV);
r.gp2 = vb_logit_pred(log_pred_X, r.ga2_w, r.ga2_V, r.ga2_invV);
r.gp3 = vb_logit_pred(log1_X, r.gi1_w, r.gi1_V, r.gi1_invV);
r.gn1 = vb_logit_pred_incr(log_pred_X, r.gf2_w, r.gf2_V, r.gf2_invV);
r.gn2 = vb_logit_pred_incr(log_pred_X, r.ga2_w, r.ga2_V, r.ga2_invV);
r.gn3 = vb_logit_pred_incr(log1_X, r.gi1_w, r.gi1_V, r.gi1_invV);
r.logdet1 = logdet(lin2_X' * lin2_X);
save('-v7', '{out}', '-struct', 'r');
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--octave', default=shutil.which('octave-cli')
                        or shutil.which('octave'),
                        help='path to the octave(-cli) executable')
    args = parser.parse_args()
    if not args.octave:
        sys.exit('Octave not found; pass its location with --octave')

    data = make_datasets()
    with tempfile.TemporaryDirectory() as tmp:
        inp = os.path.join(tmp, 'inputs.mat')
        out = os.path.join(tmp, 'outputs.mat')
        script = os.path.join(tmp, 'run_reference.m')
        savemat(inp, data)
        with open(script, 'w') as f:
            f.write(OCTAVE_SCRIPT.format(src=MATLAB_SRC, inp=inp, out=out))
        subprocess.run([args.octave, '--no-gui', '--quiet', '--no-init-file',
                        script], check=True)
        res = loadmat(out)
    res = {k: v for k, v in res.items() if not k.startswith('__')}

    version = subprocess.run([args.octave, '--version'], capture_output=True,
                             text=True).stdout.splitlines()[0]
    np.savez_compressed(OUT_FILE, octave_version=version,
                        **{'in_' + k: v for k, v in data.items()},
                        **{'out_' + k: v for k, v in res.items()})
    print(f'Wrote {len(res)} reference outputs ({version}) to {OUT_FILE}')


if __name__ == '__main__':
    main()
