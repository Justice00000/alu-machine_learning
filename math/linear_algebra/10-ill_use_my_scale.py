#!/usr/bin/env python3

def np_shape(matrix):
    """
    Return the shape of a numpy.ndarray as a tuple of integers.

    Args:
        matrix (object): Input array, which should have a 'shape' attribute.

    Returns:
        tuple: Shape of the array.

    Examples:
        >>> mat1 = [1, 2, 3, 4, 5, 6]
        >>> np_shape(mat1)
        (6,)

        >>> mat2 = []
        >>> np_shape(mat2)
        (0,)

        >>> mat3 = [[1, 2, 3], [4, 5, 6]]
        >>> np_shape(mat3)
        (2, 3)

        >>> mat4 = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
        >>> np_shape(mat4)
        (2, 2, 3)
    """
    def shape(obj):
        return (len(obj),) if isinstance(obj[0], (int, float, complex)) else (len(obj),) + shape(obj[0])

    return shape(matrix)
