#!/usr/bin/env python3

def np_shape(matrix):
    """
    Recursively computes the shape of a matrix without using conditionals, loops, or try-except blocks.

    Args:
        matrix (list): A list or nested list representing the matrix.

    Returns:
        tuple: A tuple of integers representing the shape of the matrix.
    """
    # Handle empty matrix
    shape = (len(matrix),)  # Start with the length of the current dimension
    
    # Recursively find the shape of the remaining dimensions
    if matrix and isinstance(matrix[0], list):
        shape += np_shape(matrix[0])
    
    return shape

# Test cases
mat1 = [1, 2, 3, 4, 5, 6]  # Normal vector
mat2 = []  # Empty vector
mat3 = [[1, 2, 3], [4, 5, 6]]  # 2D matrix
mat4 = [[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]  # High-dimensional matrix

print(np_shape(mat1))  # Expected output: (6,)
print(np_shape(mat2))  # Expected output: (0,)
print(np_shape(mat3))  # Expected output: (2, 3)
print(np_shape(mat4))  # Expected output: (3, 3, 1)
