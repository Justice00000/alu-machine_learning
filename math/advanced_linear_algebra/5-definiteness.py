#!/usr/bin/env python3
<<<<<<< HEAD
"""
Defines function that calculates the definiteness of a matrix
"""


=======
"""__summary__
This file contains the implementation to get the definiteness of a matrix.
"""
>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c
import numpy as np


def definiteness(matrix):
    """
<<<<<<< HEAD
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
    Comp the definiteness of a matrix.

    Args:
        matrix(numpy.ndarray): The matrix whose definiteness is to be computed.

    Returns:
        str or None :[
            "Positive definite",
            "Positive semi-definite",
            "Negative semi-definite",
            "Negative definite",
            "Indefinite",
            None
        ]
    """

    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Ensure the matrix is square (n x n) and is symmetric
    if matrix.ndim != 2 or \
        matrix.shape[0] != matrix.shape[1] or \
            np.array_equal(matrix, matrix.T) is False:
        return None

    eigenvalues = np.linalg.eigvals(matrix)

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
        print("Here")
        return None
>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c
