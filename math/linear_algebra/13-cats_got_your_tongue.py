#!/usr/bin/env python3
import numpy as np

"""
This module provides the np_cat function for concatenating numpy ndarrays
along a specified axis.

The function is designed to handle:
- 1D vectors
- 2D matrices
- High-dimensional matrices

Functionality:
- The np_cat function concatenates two numpy ndarrays along the given axis.
- It uses numpy's concatenate function and assumes that the input arrays 
  are compatible for concatenation along the specified axis.

Usage Examples:
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
        mat1 (numpy.ndarray): The first numpy ndarray to concatenate.
        mat2 (numpy.ndarray): The second numpy ndarray to concatenate.
        axis (int, optional): The axis along which the arrays will be concatenated. Default is 0.

    Returns:
        numpy.ndarray: The concatenated result of mat1 and mat2 along the specified axis.

    Raises:
        ValueError: If mat1 and mat2 cannot be concatenated along the specified axis due to mismatched dimensions.

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
