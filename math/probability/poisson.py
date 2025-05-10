#!/usr/bin/env python3
<<<<<<< HEAD

"""
This module defines a Poisson class for representing and
manipulating Poisson distributions.
"""
=======
'''
    Poisson distribution
    that represents a poisson distribution
'''
>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c


class Poisson:
    '''
<<<<<<< HEAD
    Represents a Poisson distribution.

    Attributes:
        lambtha (float): The rate (λ) of the distribution, representing
        the expected number of occurrences in a given
        time frame.
    '''

    def __init__(self, data=None, lambtha=1.):
        '''
        Initializes the Poisson distribution with data or a given λ.

        Args:
            data: List of the data to be used to estimate the distribution.
            lambtha: The expected number of occurrences in a given time frame.

        Raises:
            ValueError: If lambtha is not a positive value.
            TypeError: If data is not a list.
            ValueError: If data does not contain multiple values
        '''
        # Validate lambtha
        if lambtha <= 0:
            raise ValueError("lambtha must be a positive value")
        # Handle case when data is None (not provided)
        if data is None:
            self.lambtha = float(lambtha)
        else:
            # Validate data
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # Calculate lambtha from data
=======
        Class Poisson that represents a
        distribution of Poisson
    '''

    def factorial(self, k):
        '''
            Calculates the factorial
        '''
        if k < 0:
            return 0
        if k == 0 or k == 1:
            return 1
        return k * self.factorial(k - 1)

    def __init__(self, data=None, lambtha=1.):
        '''
            Class constructor
        '''
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        '''
<<<<<<< HEAD
        Calculates the value of the PMF for a given number of "successes".

        Args:
            k (int): The number of "successes".

        Returns:
            float: The PMF value for k.
        '''
        # Convert k to an integer if it is not
        k = int(k)
        if k < 0:
            return 0

        # Calculate the PMF using the formula
        lambda_k = self.lambtha ** k
        e_neg_lambda = self._exp(-self.lambtha)
        k_factorial = Poisson.factorial(k)

        pmf_value = (lambda_k * e_neg_lambda) / k_factorial
        return pmf_value

    def cdf(self, k):
        '''
        Calculates the value of the CDF for a given number of "successes".

        Args:
            k (int): The number of "successes".

        Returns:
            float: The CDF value for k.
        '''
        # Convert k to an integer if it is not
        k = int(k)
        if k < 0:
            return 0

        # Calculate the CDF using the formula
        cdf_value = 0
        for i in range(k + 1):
            lambda_k = self.lambtha ** i
            e_neg_lambda = self._exp(-self.lambtha)
            k_factorial = Poisson.factorial(i)
            cdf_value += (lambda_k * e_neg_lambda) / k_factorial

        return cdf_value

    def _exp(self, x):
        '''
        Calculate e^x using a series expansion.

        Args:
            x (float): The exponent.

        Returns:
            float: The value of e^x.
        '''
        e = 2.7182818285
        return e ** x

    @staticmethod
    def factorial(n):
        '''
        Calculate the factorial of n.

        Args:
            n (int): The number to calculate the factorial of.

        Returns:
            int: The factorial of n.
        '''
        if n == 0 or n == 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result
=======
            Calculates the value of the
            PMF for a given number of successes
        '''
        if k < 0:
            return 0
        k = int(k)
        e = 2.7182818285
        return ((self.lambtha ** k) * (e ** (-self.lambtha))
                ) / (self.factorial(k))

    def cdf(self, k):
        '''
            Calculates the value of the
            CDF for a given number of successes
        '''
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c
