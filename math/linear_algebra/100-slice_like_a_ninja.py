#!/usr/bin/env python3
<<<<<<< HEAD
""" defines function that slices a matrix along specific axes using numpy """


def np_slice(matrix, axes={}):
    """ returns numpy.ndarray, the slice of a matrix along specific axes """
    dimensions = len(matrix.shape)
    slices_matrix = dimensions * [slice(None)]
    for axis, value in axes.items():
        slices_matrix[axis] = slice(*value)
    return matrix[tuple(slices_matrix)]
=======
'''
Title: Slice like a Ninja | Python | Shobi
'''


def np_slice(matrix, axes={}):
    """
    Slices a matrix (numpy.ndarray) along specific axes.
    """
    slices = tuple(
        slice(*axes.get(i, (None, None, None)))
        for i in range(matrix.ndim)
    )
    return matrix[slices]
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
