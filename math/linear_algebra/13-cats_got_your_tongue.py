#!/usr/bin/env python3


"""
Module for concatenating numpy ndarrays along a specified axis.

Provides:
    - np_cat(mat1, mat2, axis=0)
"""


import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenate two numpy ndarrays along a specified axis.

    Args:
        mat1 (numpy.ndarray): First array.
        mat2 (numpy.ndarray): Second array.
        axis (int, optional): Axis along which to concatenate. Default is 0.

    Returns:
        numpy.ndarray: Concatenated array.

    Raises:
        ValueError

    Examples:
        >>> import numpy as np
        >>> mat1 = np.array([1, 2, 3])
        >>> mat2 = np.array([4, 5, 6])
        >>> np_cat(mat1, mat2)
        array([1, 2, 3, 4, 5, 6])
        >>> mat1 = np.array([[1, 2], [3, 4]])
        >>> mat2 = np.array([[5, 6]])
        >>> np_cat(mat1, mat2, axis=0)
        array([[1, 2],
               [3, 4],
               [5, 6]])
        >>> mat1 = np.array([[1, 2], [3, 4]])
        >>> mat2 = np.array([[5], [6]])
        >>> np_cat(mat1, mat2, axis=1)
        array([[1, 2, 5],
               [3, 4, 6]])
        >>> mat1 = np.array([[[1], [2]], [[3], [4]]])
        >>> mat2 = np.array([[[5], [6]]])
        >>> np_cat(mat1, mat2, axis=0)
        array([[[1],
               [2]],
              [[3],
               [4]],
              [[5],
               [6]]])
    """
    return np.concatenate((mat1, mat2), axis=axis)
