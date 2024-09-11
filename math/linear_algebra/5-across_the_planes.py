def add_matrices2D(mat1, mat2):
    """
    Adds two 2D matrices element-wise.

    Parameters:
    mat1 (list of list of int/float): The first 2D matrix.
    mat2 (list of list of int/float): The second 2D matrix.

    Returns:
    list of list of int/float or None: A new matrix representing the element-wise sum of mat1 and mat2.
    If mat1 and mat2 are not of the same shape, returns None.
    """
    # Check if the matrices have the same dimensions
    if len(mat1) != len(mat2) or any(len(row1) != len(row2) for row1, row2 in zip(mat1, mat2)):
        return None
    
    # Perform element-wise addition
    result = [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))] for i in range(len(mat1))]
    
    return result
