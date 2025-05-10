<<<<<<< HEAD
#!/usr/bin/env python3
"""Function that performs pooling on images"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Performs a convolution on images using multiple kernels
    Args:
        images: `numpy.ndarray` with shape (m, h, w)
            containing multiple grayscale images
            m: `int`, is the number of images
            h: `int`, is the height in pixels of the images
            w: `int`, is the width in pixels of the images
            c: `int`, is the number of channels in the image
        kernel_shape is a tuple of (kh, kw) containing
            the kernel shape for the pooling
            kh: `int`, is the height of the kernel
            kw: `int`, is the width of the kernel
        stride is a `tuple` of (sh, sw)
            sh: `int`, is the stride for the height of the image
            sw: `int`, is the stride for the width of the image
        mode: `str`, indicates the type of pooling
            max: indicates max pooling
            avg: indicates average pooling
    Returns:
        output: `numpy.ndarray` containing the convolved images
    """
    c = images.shape[3]
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel_shape[0], kernel_shape[1]
    sh, sw = stride[0], stride[1]
    nw = int(((w - kw) / stride[1]) + 1)
    nh = int(((h - kh) / stride[0]) + 1)
    pooled = np.zeros((m, nh, nw, c))
    for i in range(nh):
        x = i * stride[0]
        for j in range(nw):
            y = j * stride[1]
            image = images[:, x:x + kh, y:y + kw, :]
            if mode == 'max':
                pooled[:, i, j, :] = np.max(image, axis=(1, 2))
            else:
                pooled[:, i, j, :] = np.average(image, axis=(1, 2))
    return pooled
=======
#!/usr/bin/env python3
"""
Module that performs pooling operations on images
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images

    Returns:
        numpy.ndarray: Pooled images
    """
    # Get dimensions
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate output dimensions
    out_h = ((h - kh) // sh) + 1
    out_w = ((w - kw) // sw) + 1

    # Initialize output array
    output = np.zeros((m, out_h, out_w, c))

    # Perform pooling using only 2 for loops
    row = 0
    for i in range(0, h - kh + 1, sh):
        col = 0
        for j in range(0, w - kw + 1, sw):
            # Extract window
            window = images[:, i:i + kh, j:j + kw, :]

            # Apply pooling operation based on mode
            if mode == 'max':
                pool_result = np.max(window, axis=(1, 2))
            elif mode == 'avg':
                pool_result = np.mean(window, axis=(1, 2))

            output[:, row, col, :] = pool_result
            col += 1
        row += 1

    return output
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
