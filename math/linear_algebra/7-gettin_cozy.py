#!/usr/bin/env python3
'''
<<<<<<< HEAD
    This script has a function def cat_matrices2D(mat1, mat2, axis=0)
    that concatenates two matrices along a specific axis
=======
Module that concatenates two 2D matrices along a specific axis
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
'''


def cat_matrices2D(mat1, mat2, axis=0):
    '''
<<<<<<< HEAD
        The function def cat_matrices2D(mat1, mat2, axis=0)
        concatenates two matrices along a specific axis:
    '''
    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        return mat1 + mat2
    elif axis == 1 and len(mat1) == len(mat2):
        return [mat1[i] + mat2[i] for i in range(len(mat1))]
    else:
        return None
=======
    Concatenates two 2D matrices along a specific axis
    '''
    if not mat1 or not mat2:
        return None

    if axis not in [0, 1]:
        return None

    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]

    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [row1[:] + row2[:] for row1, row2 in zip(mat1, mat2)]
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
