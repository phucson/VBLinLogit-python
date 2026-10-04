import numpy as np


def logdet(A):
    """out = logdet(A)

    computes the log(det(A)) of a positive definite A.

    This function is more accurate than log(det(A))

    Copyright (c) 2013, Jan Drugowitsch
    All rights reserved.
    See the file LICENSE for licensing information.
    """
    out = 2 * np.sum(np.log(np.diag(np.linalg.cholesky(A))), 0)
    return out
