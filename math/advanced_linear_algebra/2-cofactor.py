#!/usr/bin/env python3
<<<<<<< HEAD
"""
    Defines function that calculates the cofactor matrix of a matrix
"""


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a matrix

    parameters:
        matrix [list of lists]:
            matrix whose cofactor matrix should be calculated

    returns:
        the cofactor matrix of matrix
    """
    if type(matrix) is not list:
        raise TypeError("matrix must be a list of lists")
    matrix_size = len(matrix)
    if matrix_size == 0:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if type(row) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(row) != matrix_size:
            raise ValueError("matrix must be a non-empty square matrix")
    if matrix_size == 1:
        return [[1]]
    multiplier = 1
    cofactor_matrix = []
    for row_idx in range(matrix_size):
        cofactor_row = []
        for column_idx in range(matrix_size):
            sub_matrix = []
            for row in range(matrix_size):
                if row == row_idx:
                    continue
                new_row = []
                for column in range(matrix_size):
                    if column == column_idx:
                        continue
                    new_row.append(matrix[row][column])
                sub_matrix.append(new_row)
            cofactor_row.append(multiplier * determinant(sub_matrix))
            multiplier *= -1
        cofactor_matrix.append(cofactor_row)
        if matrix_size % 2 == 0:
            multiplier *= -1
    return cofactor_matrix


def determinant(matrix):
    """
    Calculates the determinant of a matrix

    parameters:
        matrix [list of lists]:
            matrix whose determinant should be calculated

    returns:
        the determinant of matrix
    """
    if type(matrix) is not list:
        raise TypeError("matrix must be a list of lists")
    matrix_size = len(matrix)
    if matrix_size == 0:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if type(row) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(row) == 0 and matrix_size == 1:
            return 1
        if len(row) != matrix_size:
            raise ValueError("matrix must be a square matrix")
    if matrix_size == 1:
        return matrix[0][0]
    if matrix_size == 2:
        a = matrix[0][0]
        b = matrix[0][1]
        c = matrix[1][0]
        d = matrix[1][1]
        return ((a * d) - (b * c))
    multiplier = 1
    det = 0
    for i in range(matrix_size):
        element = matrix[0][i]
        sub_matrix = []
        for row in range(matrix_size):
            if row == 0:
                continue
            new_row = []
            for column in range(matrix_size):
                if column == i:
                    continue
                new_row.append(matrix[row][column])
            sub_matrix.append(new_row)
        det += (element * multiplier * determinant(sub_matrix))
        multiplier *= -1
    return det
=======
"""__summary__
This file contains the implementation to compute the cofactor matrix.
"""


def determinant(matrix):
    """Calculate the determinant of a matrix.

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


def minor(matrix):
    """Calculate the minor matrix of a given square matrix.

    Args:
        matrix (list of lists): The matrix to compute its minor matrix.

    Returns:
        list of lists: The minor matrix of the input matrix.

    Raises:
        TypeError: If the matrix is not a list of lists.
        ValueError: If the matrix is not square or is empty.
    """

    # Validate if matrix is a list of lists
    if not isinstance(matrix, list) or \
        len(matrix) == 0 or \
            not all(isinstance(row, list) for row in matrix):
        raise TypeError('matrix must be a list of lists')

    # Handle 0x0 matrix
    if (len(matrix) == 1 and len(matrix[0]) == 0):
        raise ValueError('matrix must be a non-empty square matrix')

    # Check if the matrix is square
    if not all(len(row) == len(matrix) for row in matrix):
        raise ValueError('matrix must be a non-empty square matrix')

    size = len(matrix)

    if size == 1:
        return [[1]]

    def get_minor(mat, row, col):
        """Get the minor of the matrix by removing the current row and col"""
        return [r[:col] + r[col+1:] for r in (mat[:row] + mat[row+1:])]

    # compute the minor matrix
    return [
        [
            determinant(get_minor(matrix, i, j)) for j in range(size)
        ] for i in range(size)
    ]


def cofactor(matrix):
    """_summary_

    Args:
        matrix (list of lists): The matrix to compute its cofactor.

    Returns:
        matrix: The cofactor matrix
    """
    minors = minor(matrix)
    for row in range(len(minors)):
        for col in range(len(minors[row])):
            minors[row][col] = ((-1) ** (row + col)) * minors[row][col]

    return minors
>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c
