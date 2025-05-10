<<<<<<< HEAD
#!/usr/bin/env python3
"""Function that performs a valid convolution on grayscale images"""

import numpy as np


def convolve_grayscale_same(images, kernel):
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
    pw = int(kw / 2)
    ph = int(kh / 2)
    convolved = np.zeros((m, h, w))
    npad = ((0, 0), (ph, ph), (pw, pw))
    imagesp = np.pad(images, pad_width=npad,
                     mode='constant', constant_values=0)
    for i in range(h):
        for j in range(w):
            image = imagesp[:, i:i + kh, j:j + kw]
            convolved[:, i, j] = np.sum(np.multiply(image, kernel),
                                        axis=(1, 2))
    return convolved
=======
#!/usr/bin/env python3
"""Module that performs same convolution on grayscale images"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images

    returns:
        numpy.ndarray contained convolved images
    """
    m = images.shape[0]
    height = images.shape[1]
    width = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]

    if (kh % 2) is 1:
        ph = (kh - 1) // 2
    else:
        ph = kh // 2

    if (kw % 2) is 1:
        pw = (kw - 1) // 2
    else:
        pw = kw // 2

    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                    'constant', constant_values=0)
    convoluted = np.zeros((m, height, width))

    for h in range(height):
        for w in range(width):
            output = np.sum(images[:, h:h + kh, w:w + kw] * kernel,
                            axis=1).sum(axis=1)
            convoluted[:, h, w] = output

    return convoluted
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
