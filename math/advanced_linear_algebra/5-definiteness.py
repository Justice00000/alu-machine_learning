#!/usr/bin/env python3
<<<<<<< HEAD
"""
Defines function that calculates the definiteness of a matrix
"""


=======
'''
This module contains
'''
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
import numpy as np


def definiteness(matrix):
<<<<<<< HEAD
    """
    Calculates the definiteness of a matrix

    parameters:
        matrix [numpy.ndarray of shape(n, n)]:
            matrix whose definiteness should be calculated

    returns:
        one of the following strings indicating definiteness or None:
            "Positive definite"
            "Positive semi-definite"
            "Negative definite"
            "Negative semi-definite"
            "Indefinite"
    """
    if type(matrix) is not np.ndarray:
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1] or \
       np.array_equal(matrix, matrix.T) is False:
        return None
    pos_count = 0
    neg_count = 0
    zero_count = 0
    eigenvalues = np.linalg.eig(matrix)[0]
    for value in eigenvalues:
        if value > 0:
            pos_count += 1
        if value < 0:
            neg_count += 1
        if value == 0 or value == 0.:
            zero_count += 1
    if pos_count and zero_count and neg_count == 0:
        return ("Positive semi-definite")
    elif neg_count and zero_count and pos_count == 0:
        return ("Negative semi-definite")
    elif pos_count and neg_count == 0:
        return ("Positive definite")
    elif neg_count and pos_count == 0:
        return ("Negative definite")
    return ("Indefinite")
=======
    '''
    This function determines the definiteness of a matrix.
    '''
    # Check if matrix is a numpy.ndarray
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Check if matrix is square
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    # Check if matrix is symmetric
    if not np.allclose(matrix, matrix.T):
        return None

    try:
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(matrix)
    except np.linalg.LinAlgError:
        # Matrix is not valid (e.g., contains NaN or inf)
        return None

    # Check definiteness based on eigenvalues
    if np.all(eigenvalues > 0):
        return "Positive definite"
    elif np.all(eigenvalues >= 0):
        return "Positive semi-definite"
    elif np.all(eigenvalues < 0):
        return "Negative definite"
    elif np.all(eigenvalues <= 0):
        return "Negative semi-definite"
    elif np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
        return "Indefinite"
    else:
        return None
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
