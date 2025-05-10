<<<<<<< HEAD
#!/usr/bin/env python3
""" Training with momentum
"""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """ creates the training operation for a neural network in tensorflow
        using the RMSProp optimization algorithm
        Args:
            loss: is the loss of the network
            alpha: is the learning rate
            beta2: is the RMSProp weight
            epsilon: is a small number to avoid division by zero
        Returns: the RMSProp optimization operation
    """

    optimizer = tf.train.RMSPropOptimizer(alpha, beta2, epsilon)
    return optimizer.minimize(loss)
=======
#!/usr/bin/env python3
"""
Defines function that creates the training op
for a neural network in TensorFlow using
the RMSProp optimization algorithm
"""


import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Creates the training operation for a neural network in TensorFlow
        using the RMSProp optimization algorithm

    parameters:
        loss: the loss of the network
        alpha [float]: learning rate
        beta2 [float]: RMSProp weight
        epsilon [float]: small number to avoid division by zero

    returns:
        the RMSProp optimization operation
    """
    op = tf.train.RMSPropOptimizer(
        alpha, decay=beta2, epsilon=epsilon).minimize(loss)
    return op
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
