#!/usr/bin/env python3
'''
This is a simple script that demonstrates the use of the numpy library to solve a system of linear equations.
'''
import numpy as np


def add_matrices(mat1, mat2):
    '''
    Adds two matrices together
    '''
    if np.shape(mat1) != np.shape(mat2):
        return None
    sum_mat = np.add(mat1, mat2)
    return sum_mat
