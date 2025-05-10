#!/usr/bin/env python3
<<<<<<< HEAD
"""
Defines function that calculates the inverse of a matrix
"""


def inverse(matrix):
    """
    Calculates the inverse of a matrix

    parameters:
        matrix [list of lists]:
            matrix whose inverse should be calculated

    returns:
        the inverse of matrix or None if matrix is singular
    """
    if type(matrix) is not list:
        raise TypeError("matrix must be a list of lists")
    matrix_size = len(matrix)
    for row in matrix:
        if type(row) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(row) != matrix_size or len(row) == 0:
            raise ValueError("matrix must be a non-empty square matrix")
    det = determinant(matrix)
    if det == 0:
        return None
    adjugate_matrix = adjugate(matrix)
    inverse_matrix = []
    for row_idx in range(matrix_size):
        inverse_row = []
        for column_idx in range(matrix_size):
            inverse_row.append(adjugate_matrix[row_idx][column_idx] / det)
        inverse_matrix.append(inverse_row)
    return inverse_matrix


def adjugate(matrix):
    """
    Calculates the adjugate matrix of a matrix

    parameters:
        matrix [list of lists]:
            matrix whose adjugate matrix should be calculated

    returns:
        the adjugate matrix of matrix
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
    adjugate_matrix = transpose(cofactor_matrix)
    return adjugate_matrix


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


def transpose(matrix):
    """
    Calculates the transpose of a square matrix
    Matrix is assumed to be valid and square
        based on previous type and value checks
        from prior functions in which transpose is called

    parameters:
        matrix [list of lists]:
            matrix whose transpose should be calculated

    returns:
        the transpose of matrix
    """
    matrix_size = len(matrix)
    transpose_matrix = []
    for i in range(matrix_size):
        t_row = []
        for row in range(matrix_size):
            for column in range(matrix_size):
                if column == i:
                    t_row.append(matrix[row][column])
        transpose_matrix.append(t_row)
    return transpose_matrix
=======
'''
This script demonstrates how to calculate the
inverse of a matrix without using the numpy library.
'''


def inverse(matrix):
    '''
    This function calculates the inverse of a matrix.
    '''
    if not isinstance(matrix, list) or len(matrix) == 0 or \
       not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is square
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Special case for 1x1 matrix
    if n == 1:
        if matrix[0][0] == 0:
            return None
        return [[round(1 / matrix[0][0], 1)]]

    # Create an augmented matrix [A|I]
    augmented = [row + [int(i == j) for j in range(n)]
                 for i, row in enumerate(matrix)]

    # Gaussian elimination
    for i in range(n):
        # Find pivot
        pivot = max(range(i, n), key=lambda k: abs(augmented[k][i]))
        if augmented[pivot][i] == 0:
            return None  # Matrix is singular

        # Swap rows
        augmented[i], augmented[pivot] = augmented[pivot], augmented[i]

        # Scale row
        scale = augmented[i][i]
        for j in range(i, 2*n):
            augmented[i][j] /= scale

        # Eliminate
        for k in range(n):
            if k != i:
                factor = augmented[k][i]
                for j in range(i, 2*n):
                    augmented[k][j] -= factor * augmented[i][j]

    # Extract inverse from the right half of the augmented matrix
    inverse = [row[n:] for row in augmented]

    # Format the output to match the desired precision
    formatted_inverse = []
    for row in inverse:
        formatted_row = []
        for elem in row:
            if abs(elem) < 1e-10:
                formatted_row.append(0.0)
            elif abs(elem - round(elem, 1)) < 1e-10:
                formatted_row.append(round(elem, 1))
            else:
                formatted_row.append(elem)
        formatted_inverse.append(formatted_row)

    return formatted_inverse
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
