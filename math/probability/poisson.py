#!/usr/bin/env python3
'''
Poisson Distribution
'''


class Poisson:
    '''
    Class Poisson
    '''
    def __init__(self, data=None, lambtha=1.):
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        if k < 0:
            return 0
        return (self.lambtha ** k) * \
            (self.exp(-self.lambtha) / self.factorial(k))

    def exp(self, x):
        '''
        Compute the value of e raised to the power x
        '''
        return (2.7182818285 ** x)

    def factorial(self, n):
        '''
        Factorial of a number
        '''
        if n == 0:
            return 1
        return n * self.factorial(n - 1)
