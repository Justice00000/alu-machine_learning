#!/usr/bin/env python3
<<<<<<< HEAD
"""
    A function determinant(matrix) that calculates the determinant of a matrix
=======
"""__summary__
This file contains the implementation to compute the determinant of a matrix.
>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c
"""


def determinant(matrix):
<<<<<<< HEAD
    """
    Calculates the determinant of a matrix

    Args:
        - matrix: list of lists representing the matrix

    Returns:
        - the determinant of matrix

    """
    # Check if the input is a list of lists
    if type(matrix) is not list:
        raise TypeError("matrix must be a list of lists")

    # Check if the matrix is empty
    matrix_size = len(matrix)
    if matrix_size == 0:
        raise TypeError("matrix must be a list of lists")

    # Check if the matrix is square
    for row in matrix:
        if type(row) is not list:
            raise TypeError("matrix must be a list of lists")
        if matrix == [[]]:
            return 1
        if len(row) != matrix_size:
            raise ValueError("matrix must be a square matrix")

    # Base case for 1x1 matrix
    if matrix_size == 1:
        return matrix[0][0]

    # Base case for 2x2 matrix
    if matrix_size == 2:
        a = matrix[0][0]
        b = matrix[0][1]
        c = matrix[1][0]
        d = matrix[1][1]
        return (a * d) - (b * c)

    # Calculate determinant for larger matrices
    multiplier = 1
    det = 0
    for i in range(matrix_size):
        element = matrix[0][i]
        sub_matrix = []
        for row_idx in range(1, matrix_size):
            new_row = []
            for col_idx in range(matrix_size):
                if col_idx != i:
                    new_row.append(matrix[row_idx][col_idx])
            sub_matrix.append(new_row)
        det += (element * multiplier * determinant(sub_matrix))
        multiplier *= -1
    return det
=======
    """Compute the determinant of a matrix.

    Args:
        matrix (list of lists): The matrix for which to compute its det.

    Returns:
        float: The determinant of the matrix.

    Raises:
        TypeError: If the matrix is not a list of lists.
        ValueError: If the matrix is not square or is invalid.
    """

    # Validate if matrix is a list of lists
    if not isinstance(matrix, list) or \
        len(matrix) == 0 or \
            not all(isinstance(row, list) for row in matrix):
        raise TypeError('matrix must be a list of lists')

    # Handle 0x0 matrix
    if (len(matrix) == 1 and len(matrix[0]) == 0):
        return 1

    # Check if the matrix is square
    if not all(len(row) == len(matrix) for row in matrix):
        raise ValueError('matrix must be a square matrix')

    def compute_determinant(matrix):
        n = len(matrix)
        if n == 1:
            return matrix[0][0]
        elif n == 2:
            return (
                matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])
        else:
            det = 0
            for col in range(n):
                sub_matrix = [row[:col] + row[col+1:] for row in matrix[1:]]
                cofactor = ((-1) ** col) * \
                    matrix[0][col] * compute_determinant(sub_matrix)
                det += cofactor
            return det

    return compute_determinant(matrix)
>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c
