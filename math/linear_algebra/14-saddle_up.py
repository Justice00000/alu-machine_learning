#!/usr/bin/env python3


"""
Module for performing matrix multiplication using numpy.

Provides:
    - np_matmul(mat1, mat2)
"""


import numpy as np


def np_matmul(mat1, mat2):
    """
    Perform matrix multiplication of two numpy ndarrays.

    Args:
        mat1 (numpy.ndarray): First matrix.
        mat2 (numpy.ndarray): Second matrix.

    Returns:
        numpy.ndarray: The result of multiplying mat1 by mat2.

    Raises:
        ValueError

    Examples:
        >>> import numpy as np
        >>> mat1 = np.array([[1, 2], [3, 4]])
        >>> mat2 = np.array([[5, 6], [7, 8]])
        >>> np_matmul(mat1, mat2)
        array([[19, 22],
               [43, 50]])

        >>> mat1 = np.array([[1, 0], [0, 1]])
        >>> mat2 = np.array([[2, 3], [4, 5]])
        >>> np_matmul(mat1, mat2)
        array([[2, 3],
               [4, 5]])
    """
    return np.matmul(mat1, mat2)
