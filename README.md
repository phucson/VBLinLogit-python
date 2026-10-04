# VBLinLogit

This library provides Python code, transcribed from the [MATLAB/Octave original](https://github.com/DrugowitschLab/VBLinLogit) by Jan Drugowitsch, to perform variational Bayesian linear and logistic regression. In contrast to standard linear and logistic regression, the library assumes priors over the parameters which are tuned by variational Bayesian inference, to avoid overfitting. Specifically, it supports a fully Bayesian version of automatic relevance determination (ARD), which is a sparsity-promoting prior that prunes regression coefficients that are deemed irrelevant.

Linear regression is available in the following two variants:

*   Variational Bayesian linear regression with ARD: assumes a zero-mean multivariate Gaussian prior on the weight vector, for which each element along the diagonal of the covariance matrix is modeled separately by an inverse-Gamma hyper-prior.
    
*   Variational Bayesian linear regression without ARD.

Logistic regression is available in the following two variants:

*   Variational Bayesian logistic regression with ARD: assumes a zero-mean multivariate Gaussian prior on the weight vector, for which each element along the diagonal of the covariance matrix is modeled separately by an inverse-Gamma hyper-prior.

*   Variational Bayesian logistic regression without ARD: assumes the same model as for the ARD variant, only that all elements of the diagonal covariance are modeled jointly by the same inverse-Gamma hyper-prior.

The code is licensed under the New BSD License.

## Installation

Clone the repository. To use the functions from Python, add the `src` folder to the module search path, for example by calling
```python
>>> import sys
>>> sys.path.insert(0, '/path/to/VBLinLogit-python/src')
>>> from vb_linear_fit import vb_linear_fit
```
or by setting the `PYTHONPATH` environment variable to that folder.

The installation can be checked by running the tests in the [`test`](test) folder.

## Requirements

Python 3 with [NumPy](https://numpy.org) and [SciPy](https://scipy.org). The example scripts additionally require [Matplotlib](https://matplotlib.org).

The MATLAB versions of some linear regression example scripts use the MATLAB Statistics and Machine Learning Toolbox to estimate the regression coefficient confidence intervals if it is installed. The Python versions follow the alternative branch of these scripts, and so don't plot these confidence intervals.

## Usage and documentation

The library source code resides in the [`src`](src) folder. The below provides a brief description of the API for the different functions. The header of each function file provides a more extended description of the function it performs. For a more extended discussion of the derivations and the use, please consult *Variational Bayesian
inference for linear and logistic regression*, [arxiv:1310.5438](http://arxiv.org/abs/1310.5438) [stat.ML].

See the [`examples`](examples) folder for example use of the different scripts in the `src` folder.

In all of the below, `D` is the dimensionality of the input, the output is one-dimensional, and `N` is the number of data points in the training set. For both linear and logistic regression, the training set is specified by the `N x D` NumPy array `X`, and the `N`-element one-dimensional NumPy array `y`. Vectors returned by the functions are likewise one-dimensional NumPy arrays. The `n`th row in `X` specifies one `D`-element input vector that corresponds to the output given by the `n`th element of `y`. For linear regression, these outputs are expected to be scalars. For logistic regression, they are `-1` or `1`.

### Variational Bayesian linear regression

#### Model fitting

```python
w, V, invV, logdetV, an, bn, E_a, L = vb_linear_fit(X, y)
w, V, invV, logdetV, an, bn, E_a, L = vb_linear_fit(X, y, a0, b0, c0, d0)
```
fits variational Bayesian linear regression without ARD to the training data given by `X` and `y`. The optional scalars `a0`, `b0`, `c0`, and `d0` specify the prior and hyper-prior parameters. The function returns the posterior weight mean vector `w` and covariance matrix `V`, as well as its inverse `invV` and scalar log-determinant `logdetV`. It furthermore returns the scalar posterior precision parameters, `an` and `bn`, the hyper-posterior mean `E_a`, as well as the variational bound `L`.

```python
w, V, invV, logdetV, an, bn, E_a, L = vb_linear_fit_ard(X, y)
w, V, invV, logdetV, an, bn, E_a, L = vb_linear_fit_ard(X, y, a0, b0, c0, d0)
```
is similar to `vb_linear_fit(.)`, but uses an ARD prior. Thus, it returns the hyper-posterior mean vector, `E_a`, rather than a scalar.

#### Model predictions

```python
mu, lambda_, nu = vb_linear_pred(X, w, V, an, bn)
```
for a fitted variational Bayesian linear regression model, predicts the outputs for the given `K x D` input matrix `X`, with one input vector per row. The additional arguments `w`, `V`, `an`, and `bn` are those returned by `vb_linear_fit[_ard]`. The function returns the posterior predictive means `mu`, precisions `lambda_` (`lambda` is a reserved word in Python), and degrees of freedom `nu`. `mu` and `lambda_` are `K`-element vectors, and `nu` is a scalar that is shared by all outputs.

### Variational Bayesian logistic regression

#### Model fitting

```python
w, V, invV, logdetV, E_a, L = vb_logit_fit(X, y)
w, V, invV, logdetV, E_a, L = vb_logit_fit(X, y, a0, b0)
```
fits variational Bayesian logistic regression without ARD, but a global shrinkage prior, to the training data given by `X` and `y`. The optional scalars `a0` and `b0` specify the parameters of the shrinkage prior. The function returns the posterior weight mean vector `w` and covariance matrix `V`, as well as its inverse `invV` and scalar log-determinant `logdetV`. It furthermore returns the scalar posterior shrinkage mean, `E_a`, as well as the variational bound `L`.

```python
w, V, invV, logdetV = vb_logit_fit_iter(X, y)
```
is similar to `vb_logit_fit(.)`, but uses only a weak pre-specified shrinkage prior. Thus, it does not support specifying `a0` and `b0`, and doesn't return `E_a`. Furthermore, iterates over the inputs separately rather than processing them all at once, and is therefore slower, but also computationally more stable as it avoids computing the inverse of possibly close-to-singular matrices.

```python
w, V, invV, logdetV, E_a, L = vb_logit_fit_ard(X, y)
w, V, invV, logdetV, E_a, L = vb_logit_fit_ard(X, y, a0, b0)
```
is similar to `vb_logit_fit(.)`, but uses an ARD prior. Thus, it returns the posterior shrinkage mean vector, `E_a`, rather than a scalar.

#### Model predictions

Please note that the two logistic regression prediction functions return the probabilities `p(y=1 | x, ...)` rather than the most likely `y`'s for the given inputs. How to turn these probabilities into predicted `y` depends on the loss function. For a standard `0-1` loss, the rational choice would be to predict `y=1` if `p(y=1 | x, ...) > 0.5`, and `y=-1` otherwise.

```python
out = vb_logit_pred(X, w, V, invV)
```
for a fitted variational Bayesian logistic regression model, predicts `p(y=1 | x)` for the given `K x D` input matrix `X`, with one input vector `x` per row. The additional arguments `w`, `V`, `invV`, are those returned by `vb_linear_fit[_*]`. The returned `K`-element vector contains the posterior predictive probabilities `p(y=1 | x)`, one element for each row in `X`.

```python
out = vb_logit_pred_incr(X, w, V, invV)
```
is similar to `vb_logit_pred`, but rather than computing all predictions simultaneously, it does so for each row of `X` separately by iterating over the rows of `X`.

## Original MATLAB code

This is a transcription of the MATLAB/Octave library [VBLinLogit](https://github.com/DrugowitschLab/VBLinLogit) by Jan Drugowitsch. Each `.py` file corresponds to the `.m` file of the same name.
