# VBLinLogit (Python)

A Python/NumPy port of [VBLinLogit](https://github.com/DrugowitschLab/VBLinLogit)
by Jan Drugowitsch: variational Bayesian linear and logistic regression. Unlike
standard linear and logistic regression, it puts priors on the parameters and
tunes them by variational Bayesian inference, which helps avoid overfitting. It
also supports a fully Bayesian version of automatic relevance determination
(ARD), a sparsity-promoting prior that prunes regression coefficients it deems
irrelevant.

The derivations are in *Variational Bayesian inference for linear and logistic
regression*, [arXiv:1310.5438](http://arxiv.org/abs/1310.5438) [stat.ML].

The port follows the original MATLAB code line by line. Its outputs match the
original (run under GNU Octave) to within floating-point rounding, about 1e-11
relative error (see [Verification](#verification-against-the-original-matlab-code)).

## Installation

Requires Python ≥ 3.9 with NumPy and SciPy. The examples also need Matplotlib.

```bash
pip install -e .                 # library only
pip install -e ".[examples]"     # + matplotlib for the example scripts
pip install -e ".[test]"         # + pytest
```

## Usage

```python
import numpy as np
from vblinlogit import vb_linear_fit_ard, vb_linear_pred, vb_logit_fit, vb_logit_pred

# linear regression with ARD
w, V, invV, logdetV, an, bn, E_a, L = vb_linear_fit_ard(X, y)
mu, lam, nu = vb_linear_pred(X_new, w, V, an, bn)   # Student's t predictive

# logistic regression, y in {-1, 1}
w, V, invV, logdetV, E_a, L = vb_logit_fit(X, y)
p_y1 = vb_logit_pred(X_new, w, V, invV)             # p(y = 1 | x)
```

`X` is an `N x D` array with one input per row. `y` is a length-`N` vector
(a 1-D array or an `N x 1` column both work). Vectors come back as 1-D arrays
and scalars as Python floats.

| MATLAB | Python | Returns |
|---|---|---|
| `vb_linear_fit(X, y, a0, b0, c0, d0)` | `vb_linear_fit(X, y, a0=1e-2, b0=1e-4, c0=1e-2, d0=1e-4)` | `w, V, invV, logdetV, an, bn, E_a, L` |
| `vb_linear_fit_ard(X, y, a0, b0, c0, d0)` | `vb_linear_fit_ard(...)` (same defaults) | `w, V, invV, logdetV, an, bn, E_a, L` (`E_a` is a vector) |
| `vb_linear_pred(X, w, V, an, bn)` | `vb_linear_pred(X, w, V, an, bn)` | `mu, lam, nu` |
| `vb_logit_fit(X, y, a0, b0)` | `vb_logit_fit(X, y, a0=1e-2, b0=1e-4)` | `w, V, invV, logdetV, E_a, L` |
| `vb_logit_fit_ard(X, y, a0, b0)` | `vb_logit_fit_ard(...)` (same defaults) | `w, V, invV, logdetV, E_a, L` (`E_a` is a vector) |
| `vb_logit_fit_iter(X, y)` | `vb_logit_fit_iter(X, y)` | `w, V, invV, logdetV` |
| `vb_logit_pred(X, w, V, invV)` | `vb_logit_pred(X, w, V, invV)` | `p(y=1 \| x)` per row |
| `vb_logit_pred_incr(X, w, V, invV)` | `vb_logit_pred_incr(X, w, V, invV)` | `p(y=1 \| x)` per row |
| `logdet(A)` | `logdet(A)` | `log(det(A))` via Cholesky |

Each function's docstring describes the full generative model.

MATLAB's `warning('Bayes:maxIter', ...)` becomes a
`vblinlogit.MaxIterWarning`, which you can silence with
`warnings.simplefilter('ignore', MaxIterWarning)`. MATLAB's `error(...)`
becomes a `RuntimeError`.

## Examples

The [`examples`](examples) folder contains Python versions of all eight MATLAB
example scripts:

```bash
python examples/vb_examples.py                          # run all, show figures
python examples/vb_examples.py --no-show --save-dir figs  # headless, save PNGs
python examples/vb_linear_example_sparse.py             # run one example
```

* `vb_linear_example`: VB linear regression, with and without ARD, against least squares on data with uninformative input dimensions.
* `vb_linear_example_highdim`: Bayesian shrinkage on high-dimensional data with few training examples.
* `vb_linear_example_sparse`: ARD detects and ignores irrelevant input dimensions.
* `vb_linear_example_modelsel`: model selection (polynomial order) using the variational bound.
* `vb_logit_example`: VB logistic regression, with and without ARD, against Fisher LDA.
* `vb_logit_example_coeff`: recovering coefficients and the separating hyperplane.
* `vb_logit_example_highdim`: ARD for high-dimensional logistic regression.
* `vb_logit_example_modelsel`: model selection for logistic regression using the variational bound.

## Tests

```bash
pytest                 # all tests (~40 s)
pytest -m "not slow"   # skip the two high-dimensional example runs
```

* `tests/test_linear.py`, `tests/test_logit.py`: ports of the MATLAB unit tests in `test/` (return sizes, consistency, optional arguments, weight recovery, prediction accuracy). They add checks that `vb_logit_pred` and `vb_logit_pred_incr` agree, and that `lam` handles ξ = 0.
* `tests/test_octave_parity.py`: compares every function's outputs with reference outputs of the original MATLAB code.
* `tests/test_examples.py`: runs every example script headless.

## Verification against the original MATLAB code

[`reference/matlab/src`](reference/matlab/src) holds an unmodified copy of the
original MATLAB sources. [`tools/generate_octave_reference.py`](tools/generate_octave_reference.py)
runs them under GNU Octave on fixed datasets: linear and logistic, low and
higher dimensional, with default and custom priors. It stores inputs and
outputs in `tests/data/octave_reference.npz`. `tests/test_octave_parity.py`
then checks that the Python port reproduces every returned quantity (`w`, `V`,
`invV`, `logdetV`, `an`, `bn`, `E_a`, `L`, and the predictions) with
`rtol=1e-8`. The largest relative difference observed is about 1e-11.

To regenerate the reference data (needs Octave, for example
`conda install -c conda-forge octave`):

```bash
python tools/generate_octave_reference.py --octave /path/to/octave-cli
```

## Differences from the MATLAB version

* **Random numbers.** The examples use NumPy's `default_rng`, so the generated
  datasets, and therefore the printed numbers and figures, differ from the
  MATLAB/Octave runs and the arXiv figures. Given the same data, the results
  are identical. For example, `vb_logit_example_modelsel` run on Octave's
  dataset reproduces Octave's variational bounds, losses and selected order.
  The seed of `vb_logit_example_modelsel` was changed to one where the example
  shows its intended behaviour.
* **Least squares.** MATLAB's `regress` (Statistics Toolbox), used in the
  examples for confidence intervals, is replaced by an equivalent OLS
  computation (`examples/_helpers.py:ols`). For under-determined systems
  (`N < D`, as in `vb_linear_example_sparse`), `X \ y` in MATLAB returns a
  sparse "basic" solution. NumPy's `lstsq` returns the minimum-norm solution
  instead, so the ML baseline differs there.
* **Output label.** `vb_logit_example_modelsel` printed "MSE" for what is a
  0-1 loss. The Python version labels it as 0-1 loss.

## License and credit

New BSD License; see [LICENSE](LICENSE). The original MATLAB library and the
underlying algorithms are © 2013-2019 Jan Drugowitsch. If you use this code in
research, please cite the original paper:

> J. Drugowitsch, *Variational Bayesian inference for linear and logistic
> regression*, arXiv:1310.5438 [stat.ML], 2013.
