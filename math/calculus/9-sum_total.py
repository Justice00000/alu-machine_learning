<<<<<<< HEAD
#!/usr/bin/env python3
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
#!/usr/bin/env python3
'''
Write a recursive function that takes an integer n as input and returns
the sum of the squares of the first n positive integers.
If n is not a positive integer, the function should return None.
'''


def summation_i_squared(n):
    '''
    Calculates the sum of squares from 1 to n (inclusive) recursively.

    :param n: The upper limit of the summation (inclusive)
    :return: The sum of squares, or None if n is not a valid number
    '''
    if not isinstance(n, int) or n < 1:
        return None

    if n == 1:
        return 1

    return (n * (n + 1) * (2 * n + 1)) // 6
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
