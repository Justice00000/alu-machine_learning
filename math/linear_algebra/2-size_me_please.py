#!/usr/bin/env python3
def matrix_shape(matrix):
    '''
    calculates the shape of a matrix
    '''
    ptr = matrix
    dimensions = []
    while isinstance(ptr, list):
        dimensions.append(len(ptr))
        ptr = ptr[0]
    return dimensions