#!/usr/bin/env python3
<<<<<<< HEAD
'''
    This script has a function that returns the transpose of a 2D matrix:
'''


def matrix_transpose(matrix):
    '''
        def matrix_transpose(matrix):
        returns the transpose of a 2D matrix, matrix:
    '''
    result = [
        [matrix[j][i] for j in range(len(matrix))]
        for i in range(len(matrix[0]))
    ]
    return result
=======
"""_summary_
Contains a func matrix_transpose(matrix) that returns the transpose of a matrix
"""


def matrix_transpose(matrix):
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
>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c
