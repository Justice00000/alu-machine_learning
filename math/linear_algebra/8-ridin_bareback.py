#!/usr/bin/env python3
<<<<<<< HEAD
"""_summary_
Contains a func mat_mul(mat1, mat2) that multiplies two matrices
=======
"""
This module provides a function to perform matrix multiplication.
>>>>>>> a262473352dd582d190fed95fa6e5c665f37f9ba
"""


def mat_mul(mat1, mat2):
<<<<<<< HEAD
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
        ]
    else:
        return None
=======
    """
    Performs matrix multiplication between two 2D matrices.

    Args:
        mat1 (list of lists): The first matrix to multiply.
        mat2 (list of lists): The second matrix to multiply.

    Returns:
        list of lists: The resulting matrix product.
        None: If the matrices cannot be multiplied due to dimension mismatch.
    """
    if len(mat1[0]) != len(mat2):
        return None

    result = [
        [sum(a * b for a, b in zip(row, col)) for col in zip(*mat2)]
        for row in mat1
    ]
    return result
>>>>>>> a262473352dd582d190fed95fa6e5c665f37f9ba
