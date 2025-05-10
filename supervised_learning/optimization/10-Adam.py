<<<<<<< HEAD
#!/usr/bin/env python3
""" Upgraded Adam optimization algorithm
"""


import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """ creates the training operation for a neural network in tensorflow
        using the Adam optimization algorithm
        Args:
            loss: is the loss of the network
            alpha: is the learning rate
            beta1: is the weight used for the first moment
            beta2: is the weight used for the second moment
            epsilon: is a small number to avoid division by zero
        Returns: the Adam optimization operation
    """

    optimizer = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    return optimizer.minimize(loss)
=======
#!/usr/bin/env python3
"""
Defines function that creates the training op
for a neural network in TensorFlow using
the Adam optimization algorithm
"""


import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Creates the training operation for a neural network in TensorFlow
        using the Adam optimization algorithm

    parameters:
        loss: the loss of the network
        alpha [float]: learning rate
        beta1 [float]: weight used for first moment
        beta2 [float]: weight used for second moment
        epsilon [float]: small number to avoid division by zero

    returns:
        the Adam optimization operation
    """
    op = tf.train.AdamOptimizer(
        alpha, beta1=beta1, beta2=beta2, epsilon=epsilon).minimize(loss)
    return op
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
