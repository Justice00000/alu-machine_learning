<<<<<<< HEAD
#!/usr/bin/env python3
""" Shuffle Data"""

import numpy as np


def shuffle_data(X, Y):
    """ Shuffle data points in two matrices the same way

    Args:
        X (_type_): _description_
        Y (_type_): _description_
    """
    s = np.random.permutation(X.shape[0])
    return X[s], Y[s]
=======
#!/usr/bin/env python3
"""
Defines function that shuffles the data points
in two matrices the same way
"""


import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way

    parameters:
        X [numpy.ndarray of shape (m, nx)]:
            first matrix to shuffle
            m: number of data points
            nx: the number of features in X
        Y [numpy.ndarray of shape (m, ny)]:
            second matrix to shuffle
            m: number of data points, same as in X
            ny: the number of features in Y

    returns:
        the shuffled X and Y matrices, respectively
    """
    m = X.shape[0]
    shuffle_pattern = np.random.permutation(m)
    X_shuffled = X[shuffle_pattern]
    Y_shuffled = Y[shuffle_pattern]
    return (X_shuffled, Y_shuffled)
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
