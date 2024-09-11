"""
This module provides a function to perform matrix multiplication.
"""


def mat_mul(mat1, mat2):
    """
    Performs matrix multiplication between two 2D matrices.

    Args:
        mat1 (list of lists): The first matrix to multiply.
        mat2 (list of lists): The second matrix to multiply.

    Returns:
        list of lists: The resulting matrix product.
        If the matrices cannot be multiplied due to dimension mismatch.
    """
    if len(mat1[0]) != len(mat2):
        return None

    result = [[sum(a * b for a, b in zip(row, col)) for col in zip(*mat2)] for row in mat1]
    return result
