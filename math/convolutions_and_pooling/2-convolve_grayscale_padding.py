#!/usr/bin/env python3
<<<<<<< HEAD
"""Function that performs a valid convolution
on grayscale images with custom padding"""
=======
'''
    A function def convolve_grayscale_padding(images, kernel, padding):
    that performs a same convolution on grayscale images:
'''

>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
<<<<<<< HEAD
    """Performs a convolution on grayscale images custom padding
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
        padding: `tuple` of (ph, pw)
            ph: `int` is the padding for the height of the image
            pw: `int` is the padding for the width of the image
    Returns:
        output: `numpy.ndarray` containing the convolved images
    """
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]
    ph, pw = padding[0], padding[1]
    nw = int(w - kw + (2 * pw) + 1)
    nh = int(h - kh + (2 * ph) + 1)
    convolved = np.zeros((m, nh, nw))
    npad = ((0, 0), (ph, ph), (pw, pw))
    imagesp = np.pad(images, pad_width=npad,
                     mode='constant', constant_values=0)
    for i in range(nh):
        for j in range(nw):
            image = imagesp[:, i:i + kh, j:j + kw]
            convolved[:, i, j] = np.sum(np.multiply(image, kernel),
                                        axis=(1, 2))
    return convolved
=======
    '''
        A function def convolve_grayscale_padding(images, kernel, padding):

        Args:
            images is a numpy.ndarray with shape (m, h, w)
            containing multiple grayscale images
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
            kernel is a numpy.ndarray with shape (kh, kw)
            containing the kernel for the convolution
            kh is the height of the kernel
            kw is the width of the kernel
            padding is a tuple of (ph, pw)
            ph is the padding for the height of the image
            pw is the padding for the width of the image
        Returns:
            a numpy.ndarray containing
            the convolved images
    '''
    m = images.shape[0]
    height = images.shape[1]
    width = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    ph, pw = padding
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                    'constant', constant_values=0)
    ch = height + (2 * ph) - kh + 1
    cw = width + (2 * pw) - kw + 1
    convoluted = np.zeros((m, ch, cw))
    for h in range(ch):
        for w in range(cw):
            output = np.sum(images[:, h: h + kh, w: w + kw] * kernel,
                            axis=1).sum(axis=1)
            convoluted[:, h, w] = output
    return convoluted
>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c
