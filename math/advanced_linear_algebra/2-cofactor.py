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
'''
This module contains the function to calculate the cofactor of a matrix.
'''


def cofactor(matrix):
    '''
    This function calculates the cofactor of a matrix.
    '''
    if not isinstance(matrix, list) or len(matrix) == 0\
       or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is square
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Helper function to get the submatrix
    def submatrix(matrix, i, j):
        return [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]

    # Helper function to calculate determinant
    def determinant(matrix):
        if len(matrix) == 1:
            return matrix[0][0]
        if len(matrix) == 2:
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        return sum(((-1) ** j) * matrix[0][j] *
                   determinant(submatrix(matrix, 0, j))
                   for j in range(len(matrix)))

    # Calculate the cofactor matrix
    cofactor_matrix = []
    for i in range(n):
        cofactor_row = []
        for j in range(n):
            if n == 1:
                # Special case for 1x1 matrix
                cofactor_row.append(1)
            else:
                minor = determinant(submatrix(matrix, i, j))
                cofactor_row.append((-1) ** (i + j) * minor)
        cofactor_matrix.append(cofactor_row)

    return cofactor_matrix
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
