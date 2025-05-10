#!/usr/bin/env python3
'''
<<<<<<< HEAD
    This function def np_cat(mat1, mat2, axis=0)
    concatenates two matrices along a specific axis
'''


=======
Module that concatenates two matrices along a specific axis
'''
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
import numpy as np


def np_cat(mat1, mat2, axis=0):
    '''
<<<<<<< HEAD
        Concatenate two arrays based on an axis
=======
    Concatenates two matrices along a specific axis
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
    '''
    return np.concatenate((mat1, mat2), axis=axis)
