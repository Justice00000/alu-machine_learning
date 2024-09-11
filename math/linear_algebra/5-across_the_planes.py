#!/usr/bin/env python3
"""
This module contains functions for matrix operations.
"""


def add_matrices2D(mat1, mat2):
    """
    Adds two 2D matrices element-wise.

    Args:
        mat1 (list of list of int/float): The first 2D matrix.
        mat2 (list of list of int/float): The second 2D matrix.

    Returns:
        list of list of int/float: A new matrix that is the result of element-wise addition.
        If mat1 and mat2 have different shapes, returns None.
    """

    # Check if matrices have the same dimensions
    if (len(mat1) != len(mat2) or
        any(len(row1) != len(row2) 
            for row1, row2 in zip(mat1, mat2))):
        return None

    # Add matrices element-wise
    result = [
        [mat1[i][j] + mat2[i][j]
         for j in range(len(mat1[0]))]
        for i in range(len(mat1))
    ]

    return result
