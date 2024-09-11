#!/usr/bin/env python3
def matrix_transpose(matrix):
    """
    Transposes a 2D matrix.

    Args:
        matrix (list of lists): A 2D matrix (list of lists) to be transposed.

    Returns:
        list of lists: A new matrix that is the transpose of the input matrix.
    """
    return [list(row) for row in zip(*matrix)]
