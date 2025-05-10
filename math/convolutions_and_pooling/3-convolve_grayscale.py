#!/usr/bin/env python3
<<<<<<< HEAD
"""Function that performs a valid convolution
on grayscale images with custom padding"""
=======
"""
    A function def
    convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    that performs a convolution on grayscale images:
"""

>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c

import numpy as np


<<<<<<< HEAD
def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on grayscale images
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
    kh, kw = kernel.shape[0], kernel.shape[1]
    sh, sw = stride[0], stride[1]
    if padding == 'same':
        ph = int(((w - 1) * sw + kw - w) / 2) + 1
        pw = int(((h - 1) * sh + kh - h) / 2) + 1
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        pw = padding[1]
        ph = padding[0]
    nw = int(((w - kw + (2 * pw)) / sw) + 1)
    nh = int(((h - kh + (2 * ph)) / sh) + 1)
    convolved = np.zeros((m, nh, nw))
    npad = ((0, 0), (ph, ph), (pw, pw))
    imagesp = np.pad(images, pad_width=npad,
                     mode='constant', constant_values=0)
    for i in range(nh):
        x = i * stride[0]
        for j in range(nw):
            y = j * sw
            image = imagesp[:, x:x + kh, y:y + kw]
            convolved[:, i, j] = np.sum(np.multiply(image, kernel),
                                        axis=(1, 2))
    return convolved
=======
def convolve_grayscale(images, kernel, padding="same", stride=(1, 1)):
    """
    A function def convolve_grayscale(images, kernel,
        padding='same', stride=(1, 1)):
    that performs a convolution on grayscale images:

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
        padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
        if ‘same’, performs a same convolution
        if ‘valid’, performs a valid convolution
        if a tuple:
        ph is the padding for the height of the image
        pw is the padding for the width of the image
        the image should be padded with 0’s
        stride is a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image

    Returns:
        a numpy.ndarray containing the convolved images
    """
    m = images.shape[0]
    height = images.shape[1]
    width = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    sh, sw = stride
    if padding == "same":
        ph = int(((height - 1) * stride[0] + kh - height) / 2) + 1
        pw = int(((width - 1) * stride[1] + kw - width) / 2) + 1
    elif padding == "valid":
        ph = 0
        pw = 0
    else:
        ph = padding[0]
        pw = padding[1]
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                    "constant", constant_values=0)
    ch = ((height + (2 * ph) - kh) // sh) + 1
    cw = ((width + (2 * pw) - kw) // sw) + 1
    convolved_image = np.zeros((m, ch, cw))

    i = 0
    for h in range(0, (height + (2 * ph) - kh + 1), sh):
        j = 0
        for w in range(0, (width + (2 * pw) - kw + 1), sw):
            output = np.sum(images[:, h:h + kh, w:w + kw] *
                            kernel, axis=1).sum(axis=1)
            convolved_image[:, i, j] = output
            j += 1
        i += 1
    return convolved_image
>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c
