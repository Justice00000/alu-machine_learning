#!/usr/bin/env python3
"""Module for performing convolution operations on grayscale images."""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.

    Returns:
        numpy.ndarray: The convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]

    # Calculate output dimensions for valid convolution
    output_h = h - kh + 1
    output_w = w - kw + 1

    # Initialize output array
    output = np.zeros((m, output_h, output_w))

    for i in range(m):
        # multiplication
        for j in range(output_h * output_w):
            # Convert flat index j to 2D coordinates
            row = j // output_w
            col = j % output_w

            output[i, row, col] = np.sum(
                images[i, row:row + kh, col:col + kw] * kernel
            )

    return output
