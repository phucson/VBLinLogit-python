"""VBLinLogit: variational Bayesian linear and logistic regression.

Python port of the MATLAB/Octave library by Jan Drugowitsch,
https://github.com/DrugowitschLab/VBLinLogit
"""

from ._utils import MaxIterWarning, logdet
from .linear import vb_linear_fit, vb_linear_fit_ard, vb_linear_pred
from .logit import (vb_logit_fit, vb_logit_fit_ard, vb_logit_fit_iter,
                    vb_logit_pred, vb_logit_pred_incr)

__version__ = '0.3.0'

__all__ = ['MaxIterWarning', 'logdet',
           'vb_linear_fit', 'vb_linear_fit_ard', 'vb_linear_pred',
           'vb_logit_fit', 'vb_logit_fit_ard', 'vb_logit_fit_iter',
           'vb_logit_pred', 'vb_logit_pred_incr']
