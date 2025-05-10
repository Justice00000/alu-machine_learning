#!/usr/bin/env python3
<<<<<<< HEAD
"""
Defines function that calculates the minor matrix of a matrix
"""


def minor(matrix):
    """
    Calculates the minor matrix of a matrix

    parameters:
        matrix [list of lists]:
            matrix whose minor matrix should be calculated

    returns:
        the minor matrix of matrix
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
    minor_matrix = []
    for row_idx in range(matrix_size):
        minor_row = []
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
            minor_row.append(determinant(sub_matrix))
        minor_matrix.append(minor_row)
    return minor_matrix


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
This module contains the function to calculate the minor of a matrix.
'''


def minor(matrix):
    '''
    This function calculates the minor of a matrix.
    '''
    if not isinstance(matrix, list) or len(matrix) == 0\
       or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is square
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Function to calculate the determinant of a matrix
    def determinant(mat):
        if len(mat) == 1:
            return mat[0][0]
        if len(mat) == 2:
            return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
        det = 0
        for j in range(len(mat)):
            submatrix = [row[:j] + row[j+1:] for row in mat[1:]]
            det += ((-1) ** j) * mat[0][j] * determinant(submatrix)
        return det

    # Calculate the minor matrix
    minor_matrix = []
    for i in range(n):
        minor_row = []
        for j in range(n):
            if n == 1:
                # For a 1x1 matrix, the minor is always 1
                minor_row.append(1)
            else:
                # Create submatrix by removing i-th row and j-th column
                submatrix = [row[:j] + row[j+1:] for
                             row in (matrix[:i] + matrix[i+1:])]
                minor_row.append(determinant(submatrix))
        minor_matrix.append(minor_row)

    return minor_matrix
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
