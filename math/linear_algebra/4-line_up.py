#!/usr/bin/env python3
'''
<<<<<<< HEAD
    This script has a function that adds two arrays element-wise
=======
Module that adds two arrays
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
'''


def add_arrays(arr1, arr2):
    '''
<<<<<<< HEAD
        This function adds two arrays element-wise
    '''
    if len(arr1) == len(arr2):
        return [arr1[i] + arr2[i] for i in range(len(arr1))]
    else:
        return None
=======
    Returns a new array that adds two other arrays together
    '''
    n1 = len(arr1)
    n2 = len(arr2)
    result = []

    if n1 == n2:
        for i in range(n1):
            result.append(arr1[i] + arr2[i])
        return result
    return None
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
