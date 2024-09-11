#!/usr/bin/env python3
def matrix_shape(matrix):
    """
    Calculates the shape of a matrix.

    Args:
        matrix (list of lists): A 2D matrix (list of lists).

    Returns:
        list: A list of integers representing the dimensions of the matrix.
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
