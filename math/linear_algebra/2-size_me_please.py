#!/usr/bin/env python3
<<<<<<< HEAD
'''
    This script has a function that calculates the shape of a matrix
'''


def matrix_shape(matrix):
    '''
        Calculates the shape of a matrix
    '''
    mat_shape = []
    while isinstance(matrix, list):
        mat_shape.append(len(matrix))
        matrix = matrix[0]
    return mat_shape
=======
"""
    Contains a func matrix_shape(matrix) that returns the shape of a matrix
"""


def matrix_shape(matrix):
    """_summary_

    Args:
        matrix (_type_): _description_

    Returns:
        _type_: A list of integers
    """
    if type(matrix) is not list:
        return []
    return [len(matrix)] + matrix_shape(matrix[0])
>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c
