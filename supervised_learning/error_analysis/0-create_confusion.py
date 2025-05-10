<<<<<<< HEAD
#!/usr/bin/env python3
""" Confusion matrix
"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """ creates a confusion matrix

    Args:
        labels (m, classes): correct labels in one-hot format
        logits (m, classes): predicted labels in one-hot format
    """
    m, classes = labels.shape
    confusion = np.zeros((classes, classes))
    for i in range(m):
        confusion[np.argmax(labels[i]), np.argmax(logits[i])] += 1
    return confusion
=======
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Error analysis file 1
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Function that creates a confusion matrix
    Arguments:
     - labels (numpy.ndarray): is a one-hot numpy.ndarray of shape (m, classes)
              containing the correct labels for each data point
        * m is the number of data points
        * classes is the number of classes
    - logits (numpy.ndarray): is a one-hot numpy.ndarray of shape (m, classes)
              containing the predicted labels
    Returns:
    A confusion numpy.ndarray of shape (classes, classes) with row indices
    representing the correct labels and column indices representing
    the predicted labels
    """
    return np.matmul(labels.T, logits)
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
