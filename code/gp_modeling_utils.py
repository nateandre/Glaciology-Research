""" Utility functions for GP modeling.

"""

import tensorflow as tf
from gpflow.mean_functions import MeanFunction
from gpflow.base import Parameter
from gpflow.config import default_float
import numpy as np


class Linear(MeanFunction):
    """ y_i = A * x_i + b
    """
    def __init__(self, A=None, b=None):
        """ A is a matrix which maps each element of X to Y, b is an additive constant.
        """
        MeanFunction.__init__(self)
        A = np.ones((1, 1), dtype=default_float()) if A is None else A
        b = np.zeros(1, dtype=default_float()) if b is None else b
        self.A = Parameter(np.atleast_2d(A))
        self.b = Parameter(b)

    def __call__(self, X):
        return tf.tensordot(X[:,4:], self.A, [[-1], [0]]) + self.b
