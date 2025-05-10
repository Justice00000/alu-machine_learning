#!/usr/bin/env python3
<<<<<<< HEAD

'''Module for calculating normal distribution'''
π = 3.1415926536
e = 2.7182818285


class Normal:
    """
    Represents a normal distribution.

    Attributes:
        mean (float): The mean of the distribution.
        stddev (float): The standard deviation of the distribution.
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initializes the Normal distribution with data or given mean and stddev.

        Args:
            data: List of the data to be used to estimate the distribution.
            mean: The mean of the distribution.
            stddev: The standard deviation of the distribution.

        Raises:
            ValueError: If stddev is not a positive value.
            TypeError: If data is not a list.
            ValueError: If data does not contain multiple values.
        """
        if stddev <= 0:
            raise ValueError('stddev must be a positive value')
        if data is None:
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.mean = float(sum(data) / len(data))
            self.stddev = float(
                (sum((x - self.mean) ** 2 for x in data) / len(data)) ** 0.5)

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value.

        Args:
            x: The x-value.

        Returns:
            float: The z-score of x.
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score.

        Args:
            z: The z-score.

        Returns:
            float: The x-value of z.
        """
        return self.mean + z * self.stddev

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value.

        Args:
            x: The x-value.

        Returns:
            float: The PDF value for x.
        """
        return (1.0 / (self.stddev * (2 * π) ** 0.5)) * e ** (
            -0.5 * ((x - self.mean) / self.stddev) ** 2)

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given x-value.

        Args:
            x: The x-value.

        Returns:
            float: The CDF value for x.
        """
        z = (x - self.mean) / (self.stddev * (2 ** 0.5))
        t = z - (z ** 3) / 3 + (z ** 5) / 10 - (z ** 7) / 42 + (z ** 9) / 216
        cdf = 0.5 * (1 + (2 / (π ** 0.5)) * t)
=======
'''
    Normal distribution
'''


class Normal:
    '''
        Class Normal that represents
        a normal distribution
    '''

    def __init__(self, data=None, mean=0., stddev=1.):
        '''
            Class constructor
        '''
        if data is None:
            if stddev < 1:
                raise ValueError("stddev must be a positive value")
            else:
                self.stddev = float(stddev)
                self.mean = float(mean)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                mean = float(sum(data) / len(data))
                self.mean = mean
                summation = 0
                for x in data:
                    summation += ((x - mean) ** 2)
                stddev = (summation / len(data)) ** (1 / 2)
                self.stddev = stddev

    def z_score(self, x):
        '''
            Calculates the z-score of a given x-value
        '''
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        '''
            Calculates the x-value of a given z-score
        '''
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        '''
            Calculates the value of the
            PDF for a given x-value
        '''
        mean = self.mean
        stddev = self.stddev
        e = 2.7182818285
        pi = 3.1415926536
        power = -0.5 * (self.z_score(x) ** 2)
        coefficient = 1 / (stddev * ((2 * pi) ** (1 / 2)))
        pdf = coefficient * (e ** power)
        return pdf

    def cdf(self, x):
        '''
            Calculates the value of the
            CDF for a given x-value
        '''
        mean = self.mean
        stddev = self.stddev
        pi = 3.1415926536
        value = (x - mean) / (stddev * (2 ** (1 / 2)))
        val = value - ((value ** 3) / 3) + ((value ** 5) / 10)
        val = val - ((value ** 7) / 42) + ((value ** 9) / 216)
        val *= (2 / (pi ** (1 / 2)))
        cdf = (1 / 2) * (1 + val)
>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c
        return cdf
