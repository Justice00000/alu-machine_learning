#!/usr/bin/env python3
<<<<<<< HEAD
'''
    This script has a function that adds two arrays element-wise
'''


def add_arrays(arr1, arr2):
    '''
        This function adds two arrays element-wise
    '''
    if len(arr1) == len(arr2):
        return [arr1[i] + arr2[i] for i in range(len(arr1))]
    else:
        return None
=======
"""_summary_
Contains a func add_arrays(arr1, arr2) that adds elements in same position.
"""


def add_arrays(arr1, arr2):
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
>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c
