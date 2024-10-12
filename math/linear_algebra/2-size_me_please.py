#!/usr/bin/env python3
"""
<<<<<<< HEAD
    Contains a func matrix_shape(matrix) that returns the shape of a matrix
=======
2-size_me_please - Module for calculating the shape of a matrix.

The matrix_shape function calculates the shape of a matrix and returns
it as a list of integers.
>>>>>>> a262473352dd582d190fed95fa6e5c665f37f9ba
"""


def matrix_shape(matrix):
<<<<<<< HEAD
    """_summary_

    Args:
        matrix (_type_): _description_

    Returns:
        _type_: A list of integers
    """
    if type(matrix) is not list:
        return []
    return [len(matrix)] + matrix_shape(matrix[0])
=======
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
>>>>>>> a262473352dd582d190fed95fa6e5c665f37f9ba
