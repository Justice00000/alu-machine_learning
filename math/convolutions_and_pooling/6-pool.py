#!/usr/bin/env python3
<<<<<<< HEAD
"""Function that performs pooling on images"""
=======
'''
    a function def
    pool(images, kernel_shape, pool_shape, mode='max'):
    that performs a pooling on images:
    mode: max or avg
'''

>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
<<<<<<< HEAD
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
=======
    '''
        images: numpy.ndarray with shape (m, h, w, c)
            m: number of images
            h: height in pixels
            w: width in pixels
            c: number of channels
        kernel_shape: tuple of (kh, kw)
            kh: height of the kernel
            kw: width of the kernel
        stride: tuple of (sh, sw)
            sh: stride for the height of the image
            sw: stride for the width of the image
        mode: max or avg
        Returns: numpy.ndarray containing the pooled images
    '''
    m, height, width, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    ph = ((height - kh) // sh) + 1
    pw = ((width - kw) // sw) + 1
    pooled = np.zeros((m, ph, pw, c))

    for i, h in enumerate(range(0, (height - kh + 1), sh)):
        for j, w in enumerate(range(0, (width - kw + 1), sw)):
            if mode == 'max':
                output = np.max(images[:, h:h + kh, w:w + kw, :], axis=(1, 2))
            elif mode == 'avg':
                output = np.average(images[:, h:h + kh, w:w + kw, :],
                                    axis=(1, 2))
            else:
                pass
            pooled[:, i, j, :] = output

>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c
    return pooled
