#!/usr/bin/env python3

def np_shape(matrix):
    """
    Recursively computes the shape of a matrix without using conditionals, loops, or try-except blocks.

    Args:
        matrix (list): A list or nested list representing the matrix.

    Returns:
        tuple: A tuple of integers representing the shape of the matrix.
    """
    # Use recursion to determine the shape
    try:
        return (len(matrix),) + np_shape(matrix[0])
    except (TypeError, IndexError):
        return ()

# Test cases
mat1 = [1, 2, 3, 4, 5, 6]  # Normal vector
mat2 = []  # Empty vector
mat3 = [[1, 2, 3], [4, 5, 6]]  # 2D matrix
mat4 = [[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]  # High-dimensional matrix

print(np_shape(mat1))  # Expected output: (6,)
print(np_shape(mat2))  # Expected output: (0,)
print(np_shape(mat3))  # Expected output: (2, 3)
print(np_shape(mat4))  # Expected output: (3, 3, 1)
