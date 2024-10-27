#!/usr/bin/env python3
<<<<<<< HEAD
"""_summary_
Contains a function that concatenates matrices on different axis
"""
=======


"""
Module for concatenating numpy ndarrays along a specified axis.

Provides:
    - np_cat(mat1, mat2, axis=0)
"""


>>>>>>> a262473352dd582d190fed95fa6e5c665f37f9ba
import numpy as np


def np_cat(mat1, mat2, axis=0):
<<<<<<< HEAD
    """_summary_

    Args:
        mat1 (_type_): _description_
        mat2 (_type_): _description_
        axis (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
=======
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
>>>>>>> a262473352dd582d190fed95fa6e5c665f37f9ba
    """
    return np.concatenate((mat1, mat2), axis=axis)
