#!/usr/bin/env python3

def np_shape(matrix):
    """
    Recursively computes the shape of a matrix without conditionals or loops.

    Args:
        matrix (list): A list representing the matrix.

    Returns:
        tuple: A tuple of integers representing the shape of the matrix.
    """
    return (len(matrix),) + np_shape(matrix[0]) * isinstance(matrix[0], list)
