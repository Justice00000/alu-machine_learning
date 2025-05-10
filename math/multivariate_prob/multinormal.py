<<<<<<< HEAD
#!/usr/bin/env python3
"""
Creates the MultiNormal class
"""
import numpy as np


class MultiNormal:
    """
    Represents a Multivariate Normal distribution
    """

    def __init__(self, data):
        """
        class constructor
        :param data: numpy.ndarray of shape (d, n) containing the data set:
            n is the number of data points
            d is the number of dimensions in each data point
        """
        if type(data) != np.ndarray:
            raise TypeError("data must be a 2D numpy.ndarray")
        if len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = data.mean(axis=1).reshape(d, 1)
        X_mean = data - self.mean
        self.cov = (X_mean @ X_mean.T) / (n - 1)

    def pdf(self, x):
        """
        calculates the PDF
        :param x: numpy.ndarray of shape (d, 1) containing the data point whose
        PDF should be calculated
            d is the number of dimensions of the Multinomial instance
        :return: the value of the PDF
        """
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        d = self.cov.shape[0]
        if len(x.shape) != 2 or x.shape[1] != 1 or x.shape[0] != d:
            raise ValueError("x must have the shape ({}, 1)".format(d))

        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        first = 1 / ((2 * np.pi) ** (d / 2) * np.sqrt(det))
        second = np.dot((x - self.mean).T, inv)
        third = np.dot(second, (x - self.mean) / -2)
        pdf = first * np.exp(third)

        return pdf[0][0]
=======
#!/usr/bin/env python3
'''
Computes multinormal
'''
import numpy as np


class MultiNormal:
    '''
    Represents a Multivariate Normal distribution
    '''
    def __init__(self, data):
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)

        # Calculate covariance matrix without using numpy.cov
        centered_data = data - self.mean
        self.cov = np.dot(centered_data, centered_data.T) / (n - 1)

    def pdf(self, x):
        '''
        Computes the PDF at a data point
        '''
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.cov.shape[0]
        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        # Calculate the PDF using the formula for
        # multivariate normal distribution
        det = np.linalg.det(self.cov)
        inv_cov = np.linalg.inv(self.cov)
        centered_x = x - self.mean
        exponent = -0.5 * np.dot(np.dot(centered_x.T, inv_cov), centered_x)
        normalization = 1 / ((2 * np.pi) ** (d / 2) * np.sqrt(det))
        pdf_value = normalization * np.exp(exponent)

        return float(pdf_value)
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
