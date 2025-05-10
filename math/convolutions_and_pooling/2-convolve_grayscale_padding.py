<<<<<<< HEAD
#!/usr/bin/env python3
"""Function that performs a valid convolution
on grayscale images with custom padding"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
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
#!/usr/bin/env python3
"""
Defines a function that performs convolution with custom padding
on a grayscale image
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding
    parameters:
        images [numpy.ndarray with shape (m, h, w)]:
            contains multiple grayscale images
            m: number of images
            h: height in pixels of all images
            w: width in pixels of all images
        kernel [numpy.ndarray with shape (kh, kw)]:
            contains the kernel for the convolution
            kh: height of the kernel
            kw: width of the kernel
        padding [tuple of (ph, pw)]:
            ph: padding for the height of the images
            pw: padding for the width of the images
    if needed, images should be padded with 0s
    function may only use two for loops maximum and no other loops are allowed
    returns:
        numpy.ndarray contained convolved images
    """
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
            output = np.sum(images[:, h:h + kh, w:w + kw] * kernel,
                            axis=1).sum(axis=1)
            convoluted[:, h, w] = output

    return convoluted
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
