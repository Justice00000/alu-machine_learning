#!/usr/bin/env python3
"""
2-size_me_please - Module for calculating the shape of a matrix.

The matrix_shape function calculates the shape of a matrix and returns
it as a list of integers.
"""


def matrix_shape(matrix):
    """
    Calculate the shape of a matrix.

    Args:
        matrix (list): The matrix whose shape is to be determined.

    Returns:
        list: A list of integers representing the shape of the matrix.
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
