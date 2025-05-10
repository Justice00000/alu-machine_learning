#!/usr/bin/env python3
<<<<<<< HEAD
'''
    This script has a function def mat_mul(mat1, mat2)
    that performs matrix multiplication
'''


def mat_mul(mat1, mat2):
    '''
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
=======
"""_summary_
Contains a func mat_mul(mat1, mat2) that multiplies two matrices
"""


def mat_mul(mat1, mat2):
    """_summary_

    Args:
        mat1 (_type_): _description_
        mat2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    if len(mat1[0]) == len(mat2):
        mat2_T = list(zip(*mat2))

        return [
            [
                sum(i * j for i, j in zip(row, col))
                for col in mat2_T
            ]
            for row in mat1
>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c
        ]
    else:
        return None
