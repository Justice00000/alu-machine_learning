<<<<<<< HEAD
#!/usr/bin/env python3

"""
This module defines a Poisson class for representing and
manipulating Poisson distributions.
"""


class Poisson:
    '''
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
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        '''
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
        '''
        Calculates the value of the PMF for a given number of "successes"
        '''
        if k < 0:
            return 0
        k = int(k)
        return (self.exp(-self.lambtha)
                * self.lambtha ** k) / self.factorial(k)

    def cdf(self, k):
        '''
        Calculates the value of the CDF for a given number of "successes"
        '''
        if k < 0:
            return 0
        k = int(k)
        return sum([self.pmf(n) for n in range(k + 1)])

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
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
