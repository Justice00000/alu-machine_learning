#!/usr/bin/env python3
def matrix_transpose(matrix):
    '''
    Returns s atransposed matrix based on the arguments given
    '''
    r = len(matrix)
    c = len(matrix[0])

    transposed = [[None for _ in range(r)] for _ in range(c)]
    for i in range(r):
        for j in range(c):
            transposed[j][i] = matrix[i][j]
    return transposed
