<<<<<<< HEAD
#!/usr/bin/env python3
""" Specificity
"""

import numpy as np


def specificity(confusion):
    """ calculates the specificity for each class in a confusion matrix

    Args:
        confusion (classes, classes): confusion matrix where row indices
        represent the correct labels and column indices represent the
        predicted labels

    Returns:
        (classes,): specificity of each class
    """
    true_pos = np.diag(confusion)
    false_neg = np.sum(confusion, axis=1) - true_pos
    false_pos = np.sum(confusion, axis=0) - true_pos
    true_neg = np.sum(confusion) - (true_pos + false_neg + false_pos)
    return true_neg / (true_neg + false_pos)
=======
#!/usr/bin/env python3
"""
Function specificity
"""


import numpy as np


def specificity(confusion):
    """
    Function that calculates the specificity
    for each class in a confusion matrix
    Arguments:
     - confusion is a confusion numpy.ndarray of shape (classes, classes)
            where row indices represent the correct labels and column indices
            represent the predicted labels
            classes is the number of classes
    Returns:
    A numpy.ndarray of shape (classes,) containing
    the specificity of each class
    """
    TP = np.diag(confusion)
    FN = np.sum(confusion, axis=1) - TP
    FP = np.sum(confusion, axis=0) - TP
    TN = np.sum(confusion) - (FP + FN + TP)
    SPECIFICITY = TN / (FP + TN)

    return SPECIFICITY
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
