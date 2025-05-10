#!/usr/bin/env python3
'''
<<<<<<< HEAD
    This script has a function def mat_mul(mat1, mat2)
    that performs matrix multiplication
=======
Module that multiplies two matrices
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
'''


def mat_mul(mat1, mat2):
    '''
<<<<<<< HEAD
        The function def mat_mul(mat1, mat2)
        performs matrix multiplication
    '''
    if len(mat1[0]) == len(mat2):
        return [
            [
                sum([mat1[i][k] * mat2[k][j] for k in range(len(mat1[0]))])
                for j in range(len(mat2[0]))
            ]
            for i in range(len(mat1))
        ]
    else:
        return None
=======
    Multiplies two matrices and returns the result matrix
    '''
    r1, c1 = len(mat1), len(mat1[0])
    r2, c2 = len(mat2), len(mat2[0])

    if c1 != r2:
        return None

    result_matrix = [[0 for _ in range(c2)] for _ in range(r1)]

    for i in range(r1):
        for j in range(c2):
            for k in range(c1):
                result_matrix[i][j] += mat1[i][k] * mat2[k][j]

    return result_matrix
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
