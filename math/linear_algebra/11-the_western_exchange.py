#!/usr/bin/env python3

"""
This module contains functions for performing operations on numpy ndarrays.

It includes:
- np_transpose: Transpose a numpy ndarray matrix.

Usage example:
    >>> import numpy as np
    >>> mat = np.array([[1, 2, 3], [4, 5, 6]])
    >>> np_transpose(mat)
    array([[1, 4],
           [2, 5],
           [3, 6]])
"""

def np_transpose(matrix):
    """
    Transpose a numpy.ndarray matrix.

    This function takes a numpy ndarray as input and returns its transpose. The transpose of a matrix is obtained by flipping it over its diagonal, effectively swapping the row and column indices. The function handles different dimensions of input matrices.

    Args:
        matrix (numpy.ndarray): The numpy ndarray to be transposed.

    Returns:
        numpy.ndarray: The transposed numpy ndarray.

    Examples:
        For a 2D matrix:
        >>> import numpy as np
        >>> mat = np.array([[1, 2, 3], [4, 5, 6]])
        >>> np_transpose(mat)
        array([[1, 4],
               [2, 5],
               [3, 6]])

        For a 2D row vector:
        >>> mat = np.array([[1, 2, 3]])
        >>> np_transpose(mat)
        array([[1],
               [2],
               [3]])

        For a 2D column vector:
        >>> mat = np.array([[1], [2], [3]])
        >>> np_transpose(mat)
        array([[1, 2, 3]])

        For a 1D vector:
        >>> mat = np.array([1, 2, 3])
        >>> np_transpose(mat)
        array([[1, 2, 3]])

        For a high-dimensional matrix:
        >>> mat = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        >>> np_transpose(mat)
        array([[[1, 5],
                [3, 7]],
               [[2, 6],
                [4, 8]]])
    """
    return matrix.T
