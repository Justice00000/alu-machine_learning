<<<<<<< HEAD
#!/usr/bin/env python3
""" Learing rate decay with tensorflow
"""


def batch_norm(Z, gamma, beta, epsilon):
    """ normalizes an unactivated output of a neural network using batch
    normalization

    Args:
        Z (numpy.ndarray): matrix to normalize shape (m, n)
            m: number of data points
            n: number of features
        gamma (numpy.ndarray): shape (1, n)
        contains the scales used for batch normalization
        beta (numpy.ndarray): shape (1, n)
        contains the offsets used for batch normalization
        epsilon (float): small number used to avoid division by zero

    Returns: the normalized Z matrix

    """

    mean = Z.mean(axis=0)
    variance = Z.var(axis=0)
    Z_norm = (Z - mean) / ((variance + epsilon) ** 0.5)
    Z_tilda = gamma * Z_norm + beta
    return Z_tilda
=======
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is batch normalization
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Function that normalizes an unactivated output of a NN using
    batch normalization
    Arguments:
     - Z is a numpy.ndarray of shape (m, n) that should be normalized
         * m is the number of data points
         * n is the number of features in Z
     - gamma is a numpy.ndarray of shape (1, n) containing the scales
        used for batch normalization
     - beta is a numpy.ndarray of shape (1, n) containing the offsets
        used for batch normalization
     - epsilon is a small number used to avoid division by zero
    Returns:
    The normalized Z matrix
    """

    mt = Z.mean(0)
    vt = Z.var(0)

    Zt = (Z - mt) / (vt + epsilon) ** (1/2)
    normalized_Z = gamma * Zt + beta

    return normalized_Z
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
