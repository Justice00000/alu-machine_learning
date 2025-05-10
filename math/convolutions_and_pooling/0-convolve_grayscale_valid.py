#!/usr/bin/env python3
<<<<<<< HEAD
"""Function that performs a valid convolution on grayscale images"""
=======
"""
    A function def convolve_grayscale_valid
    convolve_grayscale_valid(images, kernel)
"""

>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c

import numpy as np


def convolve_grayscale_valid(images, kernel):
<<<<<<< HEAD
    """Performs a valid convolution on grayscale images
    Args:
        images: `numpy.ndarray` with shape (m, h, w)
            containing multiple grayscale images
            m: `int`, is the number of images
            h: `int`, is the height in pixels of the images
            w: `int`, is the width in pixels of the images
        kernel: `numpy.ndarray` with shape (kh, kw)
            containing the kernel for the convolution
            kh: `int`, is the height of the kernel
            kw: `int`, is the width of the kernel
    Returns:
        output: `numpy.ndarray` containing the convolved images
    """
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]
    nw = w - kw + 1
    nh = h - kh + 1
    convolved = np.zeros((m, nh, nw))
    for i in range(nh):
        for j in range(nw):
            image = images[:, i:(i + kh), j:(j + kw)]
            convolved[:, i, j] = np.sum(np.multiply(image, kernel),
                                        axis=(1, 2))
    return convolved
=======
    """
    images is a numpy.ndarray with shape (i, y, x)
    containing multiple grayscale images
    m is the number of images
    y is the height in pixels of the images
    x is the width in pixels of the images
    kernel is a numpy.ndarray with shape (m, n)
    containing the kernel for the convolution
    kh is the height of the kernel
    kw is the width of the kernel
    Returns: a numpy.ndarray containing
    the convolved images
    """
    m, n = kernel.shape
    if m == n:
        i, y, x = images.shape
        y = y - m + 1
        x = x - m + 1
        convolved_image = np.zeros((i, y, x))
        for i in range(y):
            for j in range(x):
                shadow_area = images[:, i:i + m, j:j + n]
                convolved_image[:, i, j] = \
                    np.sum(shadow_area * kernel, axis=(1, 2))
    return convolved_image
>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c
