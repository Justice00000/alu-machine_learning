#!/usr/bin/env python3

def inverse(matrix):
    """
    Calculates the inverse of a given square matrix.

    Parameters:
    matrix (list of lists): A 2D list representing a square matrix.

    Returns:
    list of lists or None: The inverse matrix of the given matrix or None if the matrix is singular.

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
        return [r[:col] + r[col+1:] for r in (matrix[:row] + matrix[row+1:])]
    
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
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        det = 0
        for c in range(n):
            minor_matrix = minor(matrix, 0, c)
            det += ((-1) ** c) * matrix[0][c] * determinant(minor_matrix)
        return det
    
    def cofactor(matrix):
        """
        Calculates the cofactor matrix of a given square matrix.

        Parameters:
        matrix (list of lists): A 2D list representing a square matrix.

        Returns:
        list of lists: The cofactor matrix.
        """
        cofactor_matrix = []
        n = len(matrix)
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
    
    def adjugate(matrix):
        """
        Calculates the adjugate matrix of a given square matrix.

        Parameters:
        matrix (list of lists): A 2D list representing a square matrix.

        Returns:
        list of lists: The adjugate matrix.
        """
        cofactor_matrix = cofactor(matrix)
        n = len(matrix)
        # Transpose the cofactor matrix to get the adjugate matrix
        adjugate_matrix = [[cofactor_matrix[j][i] for j in range(n)] for i in range(n)]
        return adjugate_matrix

    # Check if matrix is a list of lists
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    
    # Get the size of the matrix
    n = len(matrix)
    
    # Check if the matrix is square and non-empty
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    
    # Calculate the determinant
    det = determinant(matrix)
    
    # Check if the matrix is singular
    if det == 0:
        return None
    
    # Calculate the adjugate matrix
    adjugate_matrix = adjugate(matrix)
    
    # Calculate the inverse matrix
    inverse_matrix = [[adjugate_matrix[i][j] / det for j in range(n)] for i in range(n)]
    
    return inverse_matrix
