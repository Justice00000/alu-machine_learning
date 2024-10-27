#!/usr/bin/env python3
<<<<<<< HEAD
"""__summary__
This file contains the implementation to compute the matrix of minors.
=======

"""
This module provides a function to calculate the minor matrix
of a given square matrix, along with a function to calculate
the determinant of a matrix.
>>>>>>> a262473352dd582d190fed95fa6e5c665f37f9ba
"""


def determinant(matrix):
<<<<<<< HEAD
    """Calculate the determinant of a matrix.

    Args:
        matrix (list of lists): The matrix for which to compute its det.

    Returns:
        float: The determinant of the matrix.

    Raises:
        TypeError: If the matrix is not a list of lists.
        ValueError: If the matrix is not square or is invalid.
    """

    # Validate if matrix is a list of lists
    if not isinstance(matrix, list) or \
        len(matrix) == 0 or \
            not all(isinstance(row, list) for row in matrix):
        raise TypeError('matrix must be a list of lists')

    # Handle 0x0 matrix
    if (len(matrix) == 1 and len(matrix[0]) == 0):
        return 1

    # Check if the matrix is square
    if not all(len(row) == len(matrix) for row in matrix):
        raise ValueError('matrix must be a square matrix')

    def compute_determinant(matrix):
        n = len(matrix)
        if n == 1:
            return matrix[0][0]
        elif n == 2:
            return (
                matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])
        else:
            det = 0
            for col in range(n):
                sub_matrix = [row[:col] + row[col+1:] for row in matrix[1:]]
                cofactor = ((-1) ** col) * \
                    matrix[0][col] * compute_determinant(sub_matrix)
                det += cofactor
            return det

    return compute_determinant(matrix)


def minor(matrix):
    """Calculate the minor matrix of a given square matrix.

    Args:
        matrix (list of lists): The matrix to compute its minor matrix.

    Returns:
        list of lists: The minor matrix of the input matrix.

    Raises:
        TypeError: If the matrix is not a list of lists.
        ValueError: If the matrix is not square or is empty.
    """

    # Validate if matrix is a list of lists
    if not isinstance(matrix, list) or \
        len(matrix) == 0 or \
            not all(isinstance(row, list) for row in matrix):
        raise TypeError('matrix must be a list of lists')

    # Handle 0x0 matrix
    if (len(matrix) == 1 and len(matrix[0]) == 0):
        raise ValueError('matrix must be a non-empty square matrix')

    # Check if the matrix is square
    if not all(len(row) == len(matrix) for row in matrix):
        raise ValueError('matrix must be a non-empty square matrix')

    size = len(matrix)

    if size == 1:
        return [[1]]

    def get_minor(mat, row, col):
        """Get the minor of the matrix by removing the current row and col"""
        return [r[:col] + r[col+1:] for r in (mat[:row] + mat[row+1:])]

    # compute the minor matrix
    return [
        [
            determinant(get_minor(matrix, i, j)) for j in range(size)
        ] for i in range(size)
    ]
=======
    """
    Calculates the determinant of a square matrix.

    Args:
        matrix: A list of lists representing the square matrix.

    Returns:
        The determinant of the matrix.
    """
    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return (matrix[0][0] * matrix[1][1] -
                matrix[0][1] * matrix[1][0])

    det = sum((-1) ** c * matrix[0][c] *
              determinant([row[:c] + row[c+1:] for row in matrix[1:]])
              for c in range(len(matrix)))
    return det


def minor(matrix):
    """
    Calculates the minor matrix of a given square matrix.

    Args:
        matrix: A list of lists representing the square matrix.

    Returns:
        A list of lists representing the minor matrix.

    Raises:
        TypeError: If the input is not a list of lists.
        ValueError: If the matrix is not square or is empty.
    """
    if not isinstance(matrix, list) or not all(isinstance(row, list)
                                               for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Calculate the minor matrix
    minor_matrix = []
    for i in range(n):
        minor_row = []
        for j in range(n):
            if n == 1:
                minor_row.append(1)  # For 1x1 matrix, return minor as [[1]]
            else:
                # Create a submatrix by removing the i-th row and j-th column
                submatrix = [row[:j] + row[j+1:] for row in
                             (matrix[:i] + matrix[i+1:])]
                minor_row.append(determinant(submatrix))
        minor_matrix.append(minor_row)

    return minor_matrix


# Example usage
if __name__ == '__main__':
    minor = __import__('minor').minor  # Ensure this matches your file name

    mat1 = [[5]]
    mat2 = [[1, 2], [3, 4]]
    mat3 = [[1, 1], [1, 1]]
    mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
    mat5 = []
    mat6 = [[1, 2, 3], [4, 5, 6]]

    print(minor(mat1))  # Should return [[1]]
    print(minor(mat2))  # Should return [[4, 3], [2, 1]]
    print(minor(mat3))  # Should return [[1, 1], [1, 1]]
    print(minor(mat4))  # Adjust based on your specific case

    try:
        minor(mat5)
    except Exception as e:
        print(e)

    try:
        minor(mat6)
    except Exception as e:
        print(e)
>>>>>>> a262473352dd582d190fed95fa6e5c665f37f9ba
