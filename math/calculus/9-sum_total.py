'''
Write a recursive function that takes an integer n as input and returns the sum of the squares of the first n positive integers. If n is not a positive integer, the function should return None.
'''


def summation_i_squared(n):
    '''
    Returns the sum of the squares of the first n positive integers.'''
    if not isinstance(n, int) or n < 0:
        return None
    
    if n == 0:
        return 0

    return n**2 + summation_i_squared(n - 1)
