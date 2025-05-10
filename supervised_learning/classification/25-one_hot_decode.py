#!/usr/bin/env python3
<<<<<<< HEAD

""" One-Hot Encode
=======
"""
defines function that converts a one-hot matrix
into a vector of labels
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
"""


import numpy as np


def one_hot_decode(one_hot):
<<<<<<< HEAD
    """Converts a one-hot matrix into a vector of labels

    Args:
        one_hot (_type_): _description_
    """
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None
    try:
        return np.argmax(one_hot, axis=0)
    except Exception:
        return None
=======
    """
    converts a one-hot matrix into a numeric vector of labels

    parameters:
        one-hot [numpy.ndarray with shape (classes, m)]:
            one-hot encoded matrix to decode
            classes: the maximum number of classes
            m: the number of examples
    returns:
        numpy.ndarray with shape (m,) containing the numeric labels,
            or None if fails
    """
    if type(one_hot) is not np.ndarray or len(one_hot.shape) != 2:
        return None
    vector = one_hot.transpose().argmax(axis=1)
    return vector
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
