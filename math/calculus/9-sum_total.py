#!/usr/bin/env python3
<<<<<<< HEAD
'''
    This function
    calculates the summation
    of all numbers from 1 to n
'''


def summation_i_squared(n):
    '''
    calculates the summation
    of all numbers from 1 to n
    '''
    if type(n) is not int or n < 1:
        return None
    sigma_sum = (n * (n + 1) * ((2 * n) + 1)) / 6
    return int(sigma_sum)
=======
"""
This file contains the implementation of summation_i_squared
"""


def summation_i_squared(n):
    """_summary_
    A function that computes the result a sigma notation.
    """
    if not isinstance(n, int) or n < 1:
        return None

    # Using the mathematical formular to computes the sum of squares
    return n * (n + 1) * (2 * n + 1) // 6

    # Recursive approach (Not suitable when n is large)
    # if n == 1:
    #     return 1

    # return n**2 + summation_i_squared(n-1)
>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c
