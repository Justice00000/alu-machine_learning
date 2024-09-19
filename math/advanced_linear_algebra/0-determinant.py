#!/usr/bin/env python3

def determinant(matrix):
    """
    Calculates the determinant of a square matrix.

    Parameters:
    matrix (list of lists): A 2D list representing a square matrix.

    Returns:
    int or float: The determinant of the matrix.

    Raises:
    TypeError: If the input is not a list of lists.
    ValueError: If the matrix is not square.
    """
    # Check if matrix is a list of lists
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    
    # Get the size of the matrix
    n = len(matrix)
    
    # Check if the matrix is square
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a square matrix")
    
    # Special case for 0x0 matrix
    if n == 0:
        return 1  # Can be modified to return 0 depending on the use case

    # Special case for 1x1 matrix
    if n == 1:
        return matrix[0][0]

    # Special case for 2x2 matrix
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Function to calculate the minor matrix
    def minor(matrix, row, col):
        """
        Generates the minor of the matrix by removing the specified row and column.

        Parameters:
        matrix (list of lists): The matrix from which to generate the minor.
        row (int): The row index to be removed.
        col (int): The column index to be removed.

        Returns:
        list of lists: The minor matrix.
        """
        return [r[:col] + r[col+1:] for r in (matrix[:row] + matrix[row+1:])]

    # Recursive function to calculate the determinant
    def det_recursive(matrix):
        """
        Recursively calculates the determinant of a matrix using Laplace expansion.

        Parameters:
        matrix (list of lists): The matrix for which to calculate the determinant.

        Returns:
        int or float: The determinant of the matrix.
        """
        size = len(matrix)
        if size == 2:
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

        determinant_value = 0
        for c in range(size):
            # Calculate the minor of the matrix
            minor_matrix = minor(matrix, 0, c)
            # Calculate the cofactor
            cofactor = ((-1) ** c) * matrix[0][c] * det_recursive(minor_matrix)
            determinant_value += cofactor

        return determinant_value

    return det_recursive(matrix)
