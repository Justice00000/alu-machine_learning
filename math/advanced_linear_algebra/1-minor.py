#!/usr/bin/env python3
"""
This module contains a function to calculate the minor matrix of a square matrix.

It includes:
- minor function: A function that handles matrix validation and
  calculates the minor matrix for a given square matrix.

Example usage:
    matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    print(minor(matrix))  
    # Output: [[5, 6], [8, 9]]
"""

def minor(matrix):
    """
    Calculates the minor matrix of a given square matrix.

    Parameters:
    matrix (list of lists): A 2D list representing a square matrix.

    Returns:
    list of lists: The minor matrix of the input matrix.

    Raises:
    TypeError: If the input is not a list of lists.
    ValueError: If the matrix is not a non-empty square matrix.
    """
    # Check if matrix is a list of lists
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check if the matrix is square and non-empty
    if len(matrix) == 0 or len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")

    # Get the size of the matrix
    n = len(matrix)

    # Special case for 1x1 matrix: Minor matrix is just an empty matrix
    if n == 1:
        return [[]]

    # Function to get the minor matrix by removing a specific row and column
    def get_minor(matrix, row, col):
        """
        Generates the minor matrix by removing a specific row and column.

        Parameters:
        matrix (list of lists): The matrix from which to generate the minor.
        row (int): The row index to be removed.
        col (int): The column index to be removed.

        Returns:
        list of lists: The minor matrix.
        """
        return [r[:col] + r[col+1:] for r in (matrix[:row] + matrix[row+1:])]

    # Calculate the minor matrix
    minor_matrix = []
    for i in range(n):
        minor_row = []
        for j in range(n):
            minor_row.append(get_minor(matrix, i, j))
        minor_matrix.append(minor_row)

    return minor_matrix
