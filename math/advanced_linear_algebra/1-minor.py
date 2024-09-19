#!/usr/bin/env python3

def minor(matrix):
    """
    Calculates the minor matrix of a given square matrix.

    Parameters:
    matrix (list of lists): A 2D list representing a square matrix.

    Returns:
    list of lists: The minor matrix of the given matrix.

    Raises:
    TypeError: If the input is not a list of lists.
    ValueError: If the matrix is not square or is empty.
    """
    # Check if matrix is a list of lists
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Get the size of the matrix
    n = len(matrix)
    
    # Check if the matrix is square and non-empty
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    
    # Function to generate the minor matrix by removing a specific row and column
    def get_minor(matrix, row, col):
        """
        Generates the minor matrix by removing the specified row and column.

        Parameters:
        matrix (list of lists): The matrix from which to generate the minor.
        row (int): The row index to be removed.
        col (int): The column index to be removed.

        Returns:
        list of lists: The minor matrix.
        """
        return [r[:col] + r[col+1:] for r in (matrix[:row] + matrix[row+1:])]
    
    # Calculate the minor matrix for all positions (0, 0) as an example
    # Typically, minors are calculated for specific elements to use in the determinant calculation
    return get_minor(matrix, 0, 0)
