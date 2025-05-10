#!/usr/bin/env python3
<<<<<<< HEAD

""" One-Hot Encode
=======
"""
defines function that converts a numeric label vector
into a one-hot matrix
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
"""


import numpy as np


def one_hot_encode(Y, classes):
<<<<<<< HEAD
    """Converts a numeric label vector into a one-hot matrix

    Args:
        Y (_type_): _description_
        classes (_type_): _description_
    """
    if not isinstance(Y, np.ndarray) or len(Y) == 0:
        return None
    if not isinstance(classes, int) or classes < 0:
        return None
    try:
        one_hot = np.zeros((classes, Y.shape[0]))
        one_hot[Y, np.arange(Y.shape[0])] = 1
        return one_hot
    except Exception:
=======
    """
    converts a numeric label vector into a one-hot matrix

    parameters:
        Y [numpy.ndarray with shape (m,)]: contains numeric class labels
            m is the number of examples
        classes [int]: the maximum number of classes found in Y

    returns:
        one-hot encoding of Y with shape (classes, m)
            or None if fails
    """
    # if type(Y) is not np.ndarray or len(Y.shape) != 1 or len(Y) < 1:
    # return None
    # if type(classes) is not int or classes != (Y.max() + 1):
    # return None
    # one_hot = np.eye(classes)[Y].transpose()
    # return one_hot
    if type(Y) is not np.ndarray:
        return None
    if type(classes) is not int:
        return None
    try:
        one_hot = np.eye(classes)[Y].transpose()
        return one_hot
    except Exception as err:
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
        return None
