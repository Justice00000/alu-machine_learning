<<<<<<< HEAD
#!/usr/bin/env python3
""" Sensetivity
"""

import numpy as np


def sensitivity(confusion):
    """ calculates the sensitivity for each class in a confusion matrix

    Args:
        confusion (classes, classes): confusion matrix where row indices
        represent the correct labels and column indices represent the
        predicted labels

    Returns:
        (classes,): sensitivity of each class
    """
    return np.diag(confusion) / np.sum(confusion, axis=1)
=======
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Error Analysis file 2
"""
import numpy as np


def sensitivity(confusion):
    """
    Function that calculates the sensitivity for each class
    in a confusion matrix
    Arguments:
     - confusion: is a confusion numpy.ndarray of shape (classes, classes)
                  where row indices represent the correct labels and column
                  indices represent the predicted labels
        * classes is the number of classes
    Returns:
    A numpy.ndarray of shape (classes,) containing the sensitivity
    of each class
    """
    TP = np.diag(confusion)
    FN = np.sum(confusion, axis=1) - TP
    SENSITIVITY = TP / (TP + FN)

    return SENSITIVITY
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
