#!/usr/bin/env python3
<<<<<<< HEAD
'''
    This script has a function def add_matrices2D(mat1, mat2):
    that adds two matrices element-wise
'''


def add_matrices2D(mat1, mat2):
    '''
        The function def add_matrices2D(mat1, mat2):
        adds two matrices element-wise
    '''
    if len(mat1) == len(mat2) and len(mat1[0]) == len(mat2[0]):
        return [
            [
                mat1[i][j] + mat2[i][j]
                for j in range(len(mat1[0]))
            ]
            for i in range(len(mat1))
        ]
    else:
        return None
=======
"""_summary_
Contains a func add_matrices2D(mat1, mat2) that adds 2D matrices
"""


def add_matrices2D(mat1, mat2):
    """_summary_

    Args:
        mat1 (_type_): _description_
        mat2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    output_matrix = []

    for i in range(len(mat1)):
        current_row = [
            mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))
        ]

        output_matrix.append(current_row)

    return output_matrix
>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c
