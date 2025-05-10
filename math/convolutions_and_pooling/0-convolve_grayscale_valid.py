<<<<<<< HEAD
#!/usr/bin/env python3
"""Function that performs a valid convolution on grayscale images"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
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
    height = images.shape[1]
    width = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]

    convoluted = np.zeros((m, height - kh + 1, width - kw + 1))

    for h in range(height - kh + 1):
        for w in range(width - kw + 1):
            output = np.sum(images[:, h:h + kh, w:w + kw] * kernel,
                            axis=1).sum(axis=1)
            convoluted[:, h, w] = output

    return convoluted
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
