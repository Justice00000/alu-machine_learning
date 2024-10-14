#!/usr/bin/env python3
'''
Computes multinormal
'''
import numpy as np


class MultiNormal:
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
