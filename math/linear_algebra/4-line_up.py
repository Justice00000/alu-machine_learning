#!/usr/bin/env python3
def add_arrays(arr1, arr2):
    n1 = len(arr1)
    n2 = len(arr2)
    result = []

    if n1 == n2:
        for i in range(n1):
            result.append(arr1[i] + arr2[i])
        return result
    return None
