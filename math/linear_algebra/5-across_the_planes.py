#!/usr/bin/env python3
def add_matrices2D(mat1, mat2):
    """
    Adds two 2D matrices element-wise.

    Args:
        mat1 (list of lists of int/float): The first 2D matrix.
        mat2 (list of lists of int/float): The second 2D matrix.

    Returns:
        list of lists of int/float or None: A new matrix with element-wise sums
        if matrices have the same shape, otherwise None.
    """
    # Check if matrices have the same shape
    if (len(mat1) != len(mat2)) or (any(len(row1) != len(row2) for row1, row2 in zip(mat1, mat2))):
        return None
    
    # Perform element-wise addition
    return [[a + b for a, b in zip(row1, row2)] for row1, row2 in zip(mat1, mat2)]
