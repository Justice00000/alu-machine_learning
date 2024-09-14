#!/usr/bin/env python3
'''
Module that slices a matrix along a specific axes
'''
import numpy as np

def np_slice(matrix, axes={}):
    '''
    Slices a matrix along a specific axes
    '''
    return np.s_[tuple([slice(*axes.get(i, (None, None))) for i in range(matrix.ndim)])]
