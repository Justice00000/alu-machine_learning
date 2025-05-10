#!/usr/bin/env python3
<<<<<<< HEAD
"""Function that performs a convolution on images using multiple kernels"""
=======
'''
    a function def
    convolve(images, kernels, padding='same', stride=(1, 1)):
    that performs a convolution on images using multiple kernels:
'''

>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
<<<<<<< HEAD
    """Performs a convolution on images using multiple kernels
    Args:
        images: `numpy.ndarray` with shape (m, h, w)
            containing multiple grayscale images
            m: `int`, is the number of images
            h: `int`, is the height in pixels of the images
            w: `int`, is the width in pixels of the images
            c: `int`, is the number of channels in the image
        kernels: `numpy.ndarray` with shape (kh, kw, c, nc)
            containing the kernel for the convolution
            kh: `int`, is the height of the kernel
            kw: `int`, is the width of the kernel
            nc: `int`, is the number of kernels
        padding: `tuple` of (ph, pw), ‘same’, or ‘valid’
            if `tuple`:
                ph: `int` is the padding for the height of the image
                pw: `int` is the padding for the width of the image
            if ‘same’, performs a same convolution
            if ‘valid’, performs a valid convolution
        stride is a tuple of (sh, sw)
            sh: `int`, is the stride for the height of the image
            sw: `int`, is the stride for the width of the image
    Returns:
        output: `numpy.ndarray` containing the convolved images
    """
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw, nc = kernels.shape[0], kernels.shape[1], kernels.shape[3]
    sh, sw = stride[0], stride[1]
    if padding == 'same':
        pw = int(((w - 1) * sw + kw - w) / 2) + 1
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        pw = padding[1]
        ph = padding[0]
    nw = int(((w - kw + (2 * pw)) / sw) + 1)
    nh = int(((h - kh + (2 * ph)) / sh) + 1)
    convolved = np.zeros((m, nh, nw, nc))
    npad = ((0, 0), (ph, ph), (pw, pw), (0, 0))
    imagesp = np.pad(images, pad_width=npad,
                     mode='constant', constant_values=0)
    for i in range(nh):
        x = i * sh
        for j in range(nw):
            y = j * sw
            for k in range(nc):
                image = imagesp[:, x:x + kh, y:y + kw, :]
                kernel = kernels[:, :, :, k]
                convolved[:, i, j, k] = np.sum(np.multiply(image, kernel),
                                               axis=(1, 2, 3))
    return convolved
=======
    """
    Performs a convolution on images with multiple channels
    using given padding and stride

    parameters:
        images [numpy.ndarray with shape (m, h, w, c)]:
            contains multiple images
            m: number of images
            h: height in pixels of all images
            w: width in pixels of all images
            c: number of channels in the image
        kernel [numpy.ndarray with shape (kh, kw, c)]:
            contains the kernel for the convolution
            kh: height of the kernel
            kw: width of the kernel
            c: number of channels in the image
            nc:  is the number of kernels
        padding [tuple of (ph, pw) or 'same' or 'valid']:
            ph: padding for the height of the images
            pw: padding for the width of the images
            'same' performs same convolution
            'valid' performs valid convolution
        stride [tuple of (sh, sw)]:
            sh: stride for the height of the image
            sw: stride for the width of the image

    if needed, images should be padded with 0s
    function may only use two for loops maximum and no other loops are allowed

    returns:
        numpy.ndarray contained convolved images
    """
    m, height, width, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride
    if padding is 'same':
        ph = ((((height - 1) * sh) + kh - height) // 2) + 1
        pw = ((((width - 1) * sw) + kw - width) // 2) + 1
    elif padding is 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    'constant', constant_values=0)
    ch = ((height + (2 * ph) - kh) // sh) + 1
    cw = ((width + (2 * pw) - kw) // sw) + 1
    convoluted = np.zeros((m, ch, cw, nc))

    for index in range(nc):
        kernel_index = kernels[:, :, :, index]
        for i, h in enumerate(range(0, (height + (2 * ph) - kh + 1), sh)):
            for j, w in enumerate(range(0, (width + (2 * pw) - kw + 1), sw)):
                output = np.sum(images[:, h: h + kh, w: w + kw, :]
                                * kernel_index, axis=(1, 2, 3))
                convoluted[:, i, j, index] = output

    return convoluted
>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c
