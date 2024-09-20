#!/usr/bin/env python3
'''
This script demonstrates how to calculate the
inverse of a matrix wo using the numpy library.
'''


def inverse(matrix):
    '''
    This function calculates the inverse of a matrix.
    '''
    if not isinstance(matrix, list) or len(matrix) == 0\
       or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is square
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Special case for 1x1 matrix
    if n == 1:
        if matrix[0][0] == 0:
            return None
        return [[1 / matrix[0][0]]]

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

    # Round the results to 15 decimal
    # places to avoid floating point imprecision
    return [[round(elem, 16) for elem in row]
            for row in inverse]
