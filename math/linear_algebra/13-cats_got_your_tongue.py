#!/usr/bin/env python3
<<<<<<< HEAD
'''
    This function def np_cat(mat1, mat2, axis=0)
    concatenates two matrices along a specific axis
'''


=======
"""_summary_
Contains a function that concatenates matrices on different axis
"""
>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c
import numpy as np


def np_cat(mat1, mat2, axis=0):
<<<<<<< HEAD
    '''
        Concatenate two arrays based on an axis
    '''
=======
    """_summary_

    Args:
        mat1 (_type_): _description_
        mat2 (_type_): _description_
        axis (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c
    return np.concatenate((mat1, mat2), axis=axis)
