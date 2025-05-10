#!/usr/bin/env python3
<<<<<<< HEAD
""" defines function that adds two matrices """


def matrix_shape(matrix):
    """ returns list of integers representing dimensions of given matrix """

    matrix_shape = []
    while type(matrix) is list:
        matrix_shape.append(len(matrix))
        matrix = matrix[0]
    return matrix_shape


def add_matrices(mat1, mat2):
    """ returns new matrix that is sum of two matrices added element-wise """

    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    if len(matrix_shape(mat1)) is 1:
        return [mat1[i] + mat2[i] for i in range(len(mat1))]
    return [add_matrices(mat1[i], mat2[i]) for i in range(len(mat1))]
=======
'''
This is a simple script that demonstrates the use
of the numpy library to solve a system of linear equations. | Python | Shobi
'''


def add_matrices(mat1, mat2):
    '''
    Adds two matrices together.
    '''
    if type(mat1) != type(mat2):
        return None

    if isinstance(mat1, (int, float)):
        return mat1 + mat2

    if len(mat1) != len(mat2):
        return None

    result = []

    for i in range(len(mat1)):
        if isinstance(mat1[i], list):
            sub_result = add_matrices(mat1[i], mat2[i])
            if sub_result is None:
                return None
            result.append(sub_result)
        elif isinstance(mat1[i], (int, float)):
            result.append(mat1[i] + mat2[i])
        else:
            return None

    return result
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
