#!/usr/bin/env python3
import numpy as np


"""
Module for concatenating numpy ndarrays along a specified axis.

Provides:
    - np_cat(mat1, mat2, axis=0): 
    Concatenates two numpy ndarrays along a given axis.

Usage:
    Import this module and use the np_cat function to concatenate arrays.

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


def np_cat(mat1, mat2, axis=0):
    """
    Concatenate two numpy ndarrays along a specified axis.

    Args:
        mat1 (numpy.ndarray): First array to concatenate.
        mat2 (numpy.ndarray): Second array to concatenate.
        axis (int, optional): Axis along which to concatenate. Default is 0.

    Returns:
        numpy.ndarray: Concatenated array.

    Raises:
        ValueError: 
        If mat1 and mat2 cannot be concatenated along the specified axis.

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
