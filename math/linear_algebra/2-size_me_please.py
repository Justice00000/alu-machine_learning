#!/usr/bin/env python3
'''
<<<<<<< HEAD
    This script has a function that calculates the shape of a matrix
=======
Module that calculates the shape of a matrix
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
'''


def matrix_shape(matrix):
    '''
<<<<<<< HEAD
        Calculates the shape of a matrix
    '''
    mat_shape = []
    while isinstance(matrix, list):
        mat_shape.append(len(matrix))
        matrix = matrix[0]
    return mat_shape
=======
    calculates the shape of a matrix
    '''
    ptr = matrix
    dimensions = []
    while isinstance(ptr, list):
        dimensions.append(len(ptr))
        ptr = ptr[0]
    return dimensions
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
