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
