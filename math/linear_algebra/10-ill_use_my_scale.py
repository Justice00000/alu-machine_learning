#!/usr/bin/env python3
def np_shape(matrix):
    """Return the shape of the given numpy ndarray as a tuple of integers."""
    if hasattr(matrix, 'shape'):
        return matrix.shape
    return tuple()  # Return an empty tuple if matrix does not have a shape attribute.

# Testing the function
if __name__ == "__main__":

    mat1 = np.array([1, 2, 3, 4, 5, 6])
    mat2 = np.array([])
    mat3 = np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
                     [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]])
    print(np_shape(mat1))  # Output: (6,)
    print(np_shape(mat2))  # Output: (0,)
    print(np_shape(mat3))  # Output: (2, 2, 5)
