#!/usr/bin/env python3
import numpy as np

def add_matrices2D(mat1, mat2):
    """
    Adds two 2D matrices element-wise.

    Parameters:
    mat1 (numpy.ndarray): The first matrix.
    mat2 (numpy.ndarray): The second matrix.

    Returns:
    numpy.ndarray: A new matrix representing the element-wise sum of mat1 and mat2, or None if the matrices are not the same shape.
    """
    if mat1.shape != mat2.shape:
        return None
    return mat1 + mat2
