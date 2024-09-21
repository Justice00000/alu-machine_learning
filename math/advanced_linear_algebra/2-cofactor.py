#!/usr/bin/env python3

"""
This module provides functions to calculate the cofactor matrix
and determinant of a given square matrix.
"""

def cofactor(matrix):
    """
    Calculates the cofactor matrix of a given square matrix.

    Parameters:
    matrix (list of lists): A 2D list representing a square matrix.

    Returns:
    list of lists: The cofactor matrix of the given matrix.

    Raises:
    TypeError: If the input is not a list of lists.
    ValueError: If the matrix is not square or is empty.
    """
    
    def minor(matrix, row, col):
        """
        Generates the minor matrix by removing the specified row and column.

        Parameters:
        matrix (list of lists): The matrix from which to generate the minor.
        row (int): The row index to be removed.
        col (int): The column index to be removed.

        Returns:
        list of lists: The minor matrix.
        """
        return [r[:col] + r[col + 1:] for r in 
                (matrix[:row] + matrix[row + 1:])]

    # Check if matrix is a list of lists
    if not isinstance(matrix, list) or not all(isinstance(row, list) 
                                               for row in matrix):
        raise TypeError("matrix must be a list of lists")
    
    # Get the size of the matrix
    n = len(matrix)
    
    # Check if the matrix is square and non-empty
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    def determinant(matrix):
        """
        Calculates the determinant of a square matrix.

        Parameters:
        matrix (list of lists): A 2D list representing a square matrix.

        Returns:
        int or float: The determinant of the matrix.
        """
        n = len(matrix)
        if n == 0:
            return 1
        if n == 1:
            return matrix[0][0]
        if n == 2:
            return (matrix[0][0] * matrix[1][1] - 
                    matrix[0][1] * matrix[1][0])
        
        det = 0
        for c in range(n):
            minor_matrix = minor(matrix, 0, c)
            det += ((-1) ** c) * matrix[0][c] * determinant(minor_matrix)
        return det

    # Function to calculate the cofactor matrix
    def calculate_cofactor(matrix):
        """
        Calculates the cofactor matrix.

        Parameters:
        matrix (list of lists): The matrix for which to calculate the cofactor matrix.

        Returns:
        list of lists: The cofactor matrix.
        """
        cofactor_matrix = []
        for i in range(n):
            cofactor_row = []
            for j in range(n):
                # Calculate the minor for element (i, j)
                minor_matrix = minor(matrix, i, j)
                # Calculate the cofactor
                cofactor_value = ((-1) ** (i + j)) * determinant(minor_matrix)
                cofactor_row.append(cofactor_value)
            cofactor_matrix.append(cofactor_row)
        return cofactor_matrix
    
    return calculate_cofactor(matrix)


# Example usage (if needed)
if __name__ == '__main__':
    matrix = [[1, 2], [3, 4]]
    print(cofactor(matrix))  # Example output
