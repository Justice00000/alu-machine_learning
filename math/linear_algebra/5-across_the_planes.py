#!/usr/bin/env python3
<<<<<<< HEAD
"""_summary_
Contains a func add_matrices2D(mat1, mat2) that adds 2D matrices
=======
"""
This module contains functions for matrix operations.
>>>>>>> a262473352dd582d190fed95fa6e5c665f37f9ba
"""


def add_matrices2D(mat1, mat2):
<<<<<<< HEAD
    """_summary_

    Args:
        mat1 (_type_): _description_
        mat2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    output_matrix = []

    for i in range(len(mat1)):
        current_row = [
            mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))
        ]

        output_matrix.append(current_row)

    return output_matrix
=======
    """
    Adds two 2D matrices element-wise.

    Args:
        mat1 (list of list of int/float): The first 2D matrix.
        mat2 (list of list of int/float): The second 2D matrix.

    Returns:
        A new matrix resulting from element-wise addition.
        If mat1 and mat2 have different shapes, it returns None.
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
>>>>>>> a262473352dd582d190fed95fa6e5c665f37f9ba
