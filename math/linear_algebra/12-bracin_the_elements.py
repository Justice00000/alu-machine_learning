#!/usr/bin/env python3

import numpy as np

def np_elementwise(mat1, mat2):
    """
    Perform element-wise addition, subtraction, multiplication, and division
    between two numpy ndarrays.

    Args:
        mat1 (numpy.ndarray): The first numpy ndarray.
        mat2 (numpy.ndarray): The second numpy ndarray.

    Returns:
        tuple: A tuple containing four numpy ndarrays:
            - The element-wise sum of mat1 and mat2.
            - The element-wise difference (mat1 - mat2).
            - The element-wise product (mat1 * mat2).
            - The element-wise quotient (mat1 / mat2).

    Examples:
        >>> import numpy as np
        >>> mat1 = np.array([[11, 22, 33], [44, 55, 66]])
        >>> mat2 = np.array([[1, 2, 3], [4, 5, 6]])
        >>> np_elementwise(mat1, mat2)
        (array([[12, 24, 36],
               [48, 60, 72]]),
         array([[10, 20, 30],
               [40, 50, 60]]),
         array([[ 11,  44,  99],
               [176, 275, 396]]),
         array([[11., 11., 11.],
               [11., 11., 11.]]))
    """
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2
    
    return (add, sub, mul, div)
