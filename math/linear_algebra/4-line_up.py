#!/usr/bin/env python3
<<<<<<< HEAD
"""_summary_
Contains a func add_arrays(arr1, arr2) that adds elements in same position.
=======
"""
Module for array operations.

This module provides functions to perform operations on arrays, such as
adding two arrays element-wise.

Functions:
    add_arrays(arr1, arr2): Adds two arrays element-wise.
>>>>>>> a262473352dd582d190fed95fa6e5c665f37f9ba
"""


def add_arrays(arr1, arr2):
<<<<<<< HEAD
    """_summary_

    Args:
        arr1 (int): _description_
        arr2 (int): _description_

    Returns:
        list: A new list
    """
    if len(arr1) != len(arr2):
        return None
    return [arr1[i] + arr2[i] for i in range(len(arr1))]
=======
    """
    Adds two arrays element-wise.

    Args:
        arr1 (list of int/float): The first array.
        arr2 (list of int/float): The second array.

    Returns:
        list of int/float or None: A new list with element-wise sums if arrays
        have the same shape, otherwise None.
    """
    if len(arr1) != len(arr2):
        return None
    return [a + b for a, b in zip(arr1, arr2)]
>>>>>>> a262473352dd582d190fed95fa6e5c665f37f9ba
