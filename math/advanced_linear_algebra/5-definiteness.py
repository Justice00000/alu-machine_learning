#!/usr/bin/env python3

import numpy as np

def definiteness(matrix):
    """
    Determines the definiteness of a given matrix.

    Parameters:
    matrix (numpy.ndarray): A 2D NumPy array representing a square matrix.

    Returns:
    str or None: The definiteness of the matrix ('Positive definite', 
                 'Positive semi-definite', 'Negative definite', 
                 'Negative semi-definite', 'Indefinite') or None if 
                 matrix is invalid or does not fit any category.

    Raises:
    TypeError: If the matrix is not a numpy.ndarray.
    """
    # Check if matrix is a numpy ndarray
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    
    # Check if matrix is a valid square matrix
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        return None

    # Calculate eigenvalues
    eigenvalues = np.linalg.eigvals(matrix)
    
    # Check the definiteness based on eigenvalues
    if np.all(eigenvalues > 0):
        return "Positive definite"
    elif np.all(eigenvalues >= 0):
        return "Positive semi-definite"
    elif np.all(eigenvalues < 0):
        return "Negative definite"
    elif np.all(eigenvalues <= 0):
        return "Negative semi-definite"
    else:
        return "Indefinite"
