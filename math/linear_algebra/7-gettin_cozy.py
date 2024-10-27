#!/usr/bin/env python3
<<<<<<< HEAD
"""_summary_
Contains a func cat_matrices2D() that concatenates two matrices
=======
"""
This module provides a function to concatenate two 2D matrices.
>>>>>>> a262473352dd582d190fed95fa6e5c665f37f9ba
"""


def cat_matrices2D(mat1, mat2, axis=0):
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
    Concatenates two 2D matrices along a specific axis.

    Args:
        mat1 (list of lists): The first 2D matrix to concatenate.
        mat2 (list of lists): The second 2D matrix to concatenate.
        axis (int, optional): The axis along which to concatenate.

    Returns:
        A new 2D matrix resulting from concatenation if dimensions match.
        If the matrices cannot be concatenated due to mismatched dimensions.
>>>>>>> a262473352dd582d190fed95fa6e5c665f37f9ba
    """
    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        return mat1 + mat2
    elif axis == 1 and len(mat1) == len(mat2):
<<<<<<< HEAD
        return [mat1[i] + mat2[i] for i in range(len(mat1))]
    else:
        return None
=======
        return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
    return None
>>>>>>> a262473352dd582d190fed95fa6e5c665f37f9ba
