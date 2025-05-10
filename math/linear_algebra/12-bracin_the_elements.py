#!/usr/bin/env python3
<<<<<<< HEAD
'''
    This script has a function def np_elementwise(mat1, mat2)
    that performs element-wise addition,
    subtraction, multiplication, and division:
'''


def np_elementwise(mat1, mat2):
    '''
        def np_elementwise(mat1, mat2)
        that performs element-wise addition,
        subtraction, multiplication, and division:
    '''
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
=======
"""_summary_
Contains a func np_elementwise(mat1, mat2) that performs
element-wise operations on two matrices
"""


def np_elementwise(mat1, mat2):
    """_summary_

    Args:
        mat1 (_type_): _description_
        mat2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    return [mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2]
>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c
