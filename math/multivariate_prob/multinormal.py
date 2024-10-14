#!/usr/bin/env python3
'''
Computes multinormal
'''
import numpy as np


class MultiNormal:
    '''
    Class that represents a Multivariate Normal distribution
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
        Calculates the PDF at a data point
        '''
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d, _ = self.mean.shape
        if x.shape != (d, 1):
            raise ValueError(f"x must have the shape ({d}, 1)")

        # Calculate the PDF
        det = np.linalg.det(self.cov)
        inv_cov = np.linalg.inv(self.cov)
        x_centered = x - self.mean
        exponent = -0.5 * np.dot(np.dot(x_centered.T, inv_cov), x_centered)
        coefficient = 1 / np.sqrt((2 * np.pi) ** d * det)

        return float(coefficient * np.exp(exponent))
