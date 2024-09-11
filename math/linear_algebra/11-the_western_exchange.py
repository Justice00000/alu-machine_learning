#!/usr/bin/env python3
import numpy as np

def np_transpose(matrix: np.ndarray) -> np.ndarray:
    """
    Transpose a numpy.ndarray matrix.

    This function takes a numpy ndarray as input and returns its transpose. The transpose of a matrix is obtained by flipping it over its diagonal, effectively swapping the row and column indices.

    Args:
        matrix (np.ndarray): The numpy ndarray to be transposed.

    Returns:
        np.ndarray: The transposed numpy ndarray.

    Examples:
        >>> import numpy as np
        >>> mat = np.array([[1, 2, 3], [4, 5, 6]])
        >>> np_transpose(mat)
        array([[1, 4],
               [2, 5],
               [3, 6]])
    """
    return matrix.T
