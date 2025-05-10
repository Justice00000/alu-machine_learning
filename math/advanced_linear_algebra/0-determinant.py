#!/usr/bin/env python3
<<<<<<< HEAD
"""
    A function determinant(matrix) that calculates the determinant of a matrix
"""


def determinant(matrix):
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
'''
This module contains a function that calculates the determinant of a matrix.
'''


def determinant(matrix):
    '''
    This function calculates the determinant of a matrix.
    '''
    if not isinstance(matrix, list) or len(matrix) == 0 \
       or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Handle 0x0 matrix
    if matrix == [[]]:
        return 1

    # Check if matrix is square
    matrix_size = len(matrix)
    if any(len(row) != matrix_size for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # Base case: 1x1 matrix
    if matrix_size == 1:
        return matrix[0][0]

    # Base case: 2x2 matrix
    if matrix_size == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Recursive case: nxn matrix
    det_value = 0
    for col in range(matrix_size):
        # Create submatrix by removing first row and current column
        submatrix = [row[:col] + row[col+1:] for row in matrix[1:]]
        # Recursive call and accumulate determinant
        cofactor = (-1) ** col
        det_value += cofactor * matrix[0][col] * determinant(submatrix)

    return det_value
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
