#!/usr/bin/env python3
<<<<<<< HEAD
"""_summary_
Contains a func matrix_transpose(matrix) that returns the transpose of a matrix
=======
"""
Module for matrix operations.

This module provides functions to perform operations on matrices, such as
transposing a 2D matrix.

Functions:
    matrix_transpose(matrix): Returns the transpose of the provided 2D matrix.
>>>>>>> a262473352dd582d190fed95fa6e5c665f37f9ba
"""


def matrix_transpose(matrix):
<<<<<<< HEAD
    """_summary_

    Args:
        matrix (_type_): _description_

    Returns:
        _type_: A list of lists
    """
    new_matrix = [
        [0 for _ in range(len(matrix))] for _ in range(len(matrix[0]))
    ]
    for r in range(len(matrix)):
        for c in range(len(matrix[r])):
            new_matrix[c][r] = matrix[r][c]
    return new_matrix
=======
    """
    Transposes a 2D matrix.

    Args:
        matrix (list of lists): A 2D matrix (list of lists) to be transposed.

    Returns:
        list of lists: A new matrix that is the transpose of the input matrix.
    """
    return [list(row) for row in zip(*matrix)]
>>>>>>> a262473352dd582d190fed95fa6e5c665f37f9ba
