import numpy as np

def determinant(matrix):
    # Check if matrix is a list of lists
    if not all(isinstance(row, list) for row in matrix) or not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    
    # Check if the matrix is square
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a square matrix")
    
    # Special cases for 0x0, 1x1, 2x2, and 3x3 matrices
    if n == 0:
        return 1  # By definition, the determinant of a 0x0 matrix is 1
    
    if n == 1:
        return matrix[0][0]
    
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    if n == 3:
        return (matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
                matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
                matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]))
    
    # General case using recursion
    def minor(matrix, row, col):
        return [r[:col] + r[col+1:] for r in (matrix[:row] + matrix[row+1:])]

    def det_recursive(matrix):
        size = len(matrix)
        if size == 2:
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        determinant_value = 0
        for c in range(size):
            determinant_value += ((-1) ** c) * matrix[0][c] * det_recursive(minor(matrix, 0, c))
        return determinant_value
    
    return det_recursive(matrix)

