#!/usr/bin/env python3
"""
This module provides a function to concatenate two 2D matrices.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two 2D matrices along a specific axis.

    Args:
        mat1 (list of lists): The first 2D matrix to concatenate.
        mat2 (list of lists): The second 2D matrix to concatenate.
        axis (int, optional): The axis along which to concatenate. Defaults to 0.

    Returns:
        A new 2D matrix resulting from concatenation if dimensions match.
        None: If the matrices cannot be concatenated due to mismatched dimensions.
    """
    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        return mat1 + mat2
    elif axis == 1 and len(mat1) == len(mat2):
        return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
    return None
