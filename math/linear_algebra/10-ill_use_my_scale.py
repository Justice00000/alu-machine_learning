#!/usr/bin/env python3

def np_shape(matrix):
    """
    Recursively computes the shape of a matrix.

    Args:
        matrix (list): A list representing the matrix.

    Returns:
        tuple: A tuple of integers representing the shape of the matrix.
    """
    # Base cases when matrix is empty or non-list
    return (len(matrix),) + np_shape(matrix[0])
    if matrix and isinstance(matrix[0], list) else (len(matrix),)


# Test cases to validate correctness
if __name__ == "__main__":
    mat1 = [1, 2, 3, 4, 5, 6]
    mat2 = []
    mat3 = [[1, 2, 3], [4, 5, 6]]
    mat4 = [[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]
    
    print(np_shape(mat1))  # Output: (6,)
    print(np_shape(mat2))  # Output: (0,)
    print(np_shape(mat3))  # Output: (2, 3)
    print(np_shape(mat4))  # Output: (3, 3, 1)
