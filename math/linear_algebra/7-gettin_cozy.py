#!/usr/bin/env python3
def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two 2D matrices along a specified axis.

    Args:
        mat1 (list of lists of int/float): The first 2D matrix.
        mat2 (list of lists of int/float): The second 2D matrix.
        axis (int): The axis along which to concatenate (0 for rows, 1 for columns).

    Returns:
        list of lists of int/float or None: A new matrix resulting from concatenation,
        or None if the matrices cannot be concatenated.
    """
    if axis == 0:
        # Check if number of columns is the same
        if len(mat1[0]) != len(mat2[0]):
            return None
        return mat1 + mat2
    elif axis == 1:
        # Check if number of rows is the same
        if len(mat1) != len(mat2):
            return None
        return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
    else:
        return None
