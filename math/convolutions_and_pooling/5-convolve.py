<<<<<<< HEAD
#!/usr/bin/env python3
"""Function that performs a convolution on images using multiple kernels"""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
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
#!/usr/bin/env python3
"""
Module that performs convolution on images using multiple kernels
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs convolution on images using multiple kernels

    Returns:
        numpy.ndarray: Convolved images
    """
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
    sh, sw = stride

    # Calculate padding
    if padding == 'same':
        ph = ((((h - 1) * sh) + kh - h) // 2) + 1
        pw = ((((w - 1) * sw) + kw - w) // 2) + 1
    elif padding == 'valid':
        ph = pw = 0
    else:
        ph, pw = padding

    # Pad images
    padded = np.pad(images,
                    ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    mode='constant',
                    constant_values=0)

    # Calculate output dimensions
    out_h = ((h + (2 * ph) - kh) // sh) + 1
    out_w = ((w + (2 * pw) - kw) // sw) + 1

    # Initialize output array
    output = np.zeros((m, out_h, out_w, nc))

    # Perform convolution using only 3 for loops
    for k in range(nc):  # Loop through each kernel
        curr_kernel = kernels[:, :, :, k]
        i = 0
        for row in range(0, h + (2 * ph) - kh + 1, sh):  # Height stride
            j = 0
            for col in range(0, w + (2 * pw) - kw + 1, sw):  # Width stride
                # Extract window and perform element-wise multiplication
                window = padded[:, row:row + kh, col:col + kw, :]
                # Sum across all dimensions except batch (m)
                conv_result = np.sum(window * curr_kernel, axis=(1, 2, 3))
                output[:, i, j, k] = conv_result
                j += 1
            i += 1

    return output
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
