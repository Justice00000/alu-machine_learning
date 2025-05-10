#!/usr/bin/env python3
'''
<<<<<<< HEAD
    This script has a function that returns the transpose of a 2D matrix:
=======
This module contains a function that returns a transposed matrix
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
'''


def matrix_transpose(matrix):
    '''
<<<<<<< HEAD
        def matrix_transpose(matrix):
        returns the transpose of a 2D matrix, matrix:
    '''
    result = [
        [matrix[j][i] for j in range(len(matrix))]
        for i in range(len(matrix[0]))
    ]
    return result
=======
    Returns s atransposed matrix based on the arguments given
    '''
    r = len(matrix)
    c = len(matrix[0])

    transposed = [[None for _ in range(r)] for _ in range(c)]
    for i in range(r):
        for j in range(c):
            transposed[j][i] = matrix[i][j]
    return transposed
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
