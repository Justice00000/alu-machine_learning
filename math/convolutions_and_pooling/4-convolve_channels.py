<<<<<<< HEAD
#!/usr/bin/env python3
"""Function that performs a performs a convolution on images with channels"""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on images with channels
    Args:
        images: `numpy.ndarray` with shape (m, h, w)
            containing multiple grayscale images
            m: `int`, is the number of images
            h: `int`, is the height in pixels of the images
            w: `int`, is the width in pixels of the images
            c: `int`, is the number of channels in the image
        kernel: `numpy.ndarray` with shape (kh, kw, c)
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
    convolved = np.zeros((m, nh, nw))
    npad = ((0, 0), (ph, ph), (pw, pw), (0, 0))
    imagesp = np.pad(images, pad_width=npad,
                     mode='constant', constant_values=0)
    for i in range(nh):
        x = i * sh
        for j in range(nw):
            y = j * sw
            image = imagesp[:, x:x + kh, y:y + kw, :]
            convolved[:, i, j] = np.sum(np.multiply(image, kernel),
                                        axis=(1, 2, 3))
    return convolved
=======
#!/usr/bin/env python3
"""
Defines a function that performs convolution
on a image with multiple channels using given padding and stride
"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
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
        padding [tuple of (ph, pw) or 'same' or 'valid']:
            ph: padding for the height of the images
            pw: padding for the width of the images
            'same' performs same convolution
            'valid' performs valid convoltuion
        stride [tuple of (sh, sw)]:
            sh: stride for the height of the image
            sw: stride for the width of the image
    if needed, images should be padded with 0s
    function may only use two for loops maximum and no other loops are allowed
    returns:
        numpy.ndarray contained convolved images
    """
    m, height, width, c = images.shape
    kh, kw, kc = kernel.shape
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

    convoluted = np.zeros((m, ch, cw))

    i = 0
    for h in range(0, (height + (2 * ph) - kh + 1), sh):
        j = 0
        for w in range(0, (width + (2 * pw) - kw + 1), sw):
            output = np.sum(images[:, h:h + kh, w:w + kw, :] * kernel,
                            axis=1).sum(axis=1).sum(axis=1)
            convoluted[:, i, j] = output
            j += 1
        i += 1

    return convoluted
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
