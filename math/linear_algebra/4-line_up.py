#!/usr/bin/env python3
"""
Module for array operations.

This module provides functions to perform operations on arrays, such as
adding two arrays element-wise.

Functions:
    add_arrays(arr1, arr2): Adds two arrays element-wise.
"""


def add_arrays(arr1, arr2):
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
