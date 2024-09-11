#!/usr/bin/env python3
def mat_mul(mat1, mat2):
    """
    Multiplies two 2D matrices.

    Args:
        mat1 (list of lists of int/float): The first 2D matrix.
        mat2 (list of lists of int/float): The second 2D matrix.

    Returns:
        list of lists of int/float or None: A new matrix resulting from the multiplication,
        or None if the matrices cannot be multiplied.
    """
    # Check if multiplication is possible
    if len(mat1[0]) != len(mat2):
        return None

    # Initialize the result matrix with zeroes
    result = [[0] * len(mat2[0]) for _ in range(len(mat1))]

    # Perform matrix multiplication
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            result[i][j] = sum(mat1[i][k] * mat2[k][j] for k in range(len(mat2)))
    
    return result
