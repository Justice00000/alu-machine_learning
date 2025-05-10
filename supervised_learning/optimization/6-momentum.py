<<<<<<< HEAD
#!/usr/bin/env python3
""" Training with momentum
"""

import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """ creates the training operation for a neural network in tensorflow
        using the gradient descent with momentum optimization algorithm
        Args:
            loss: is the loss of the network
            alpha: is the learning rate
            beta1: is the momentum weight
        Returns: the momentum optimization operation
    """

    optimizer = tf.train.MomentumOptimizer(alpha, beta1)
    return optimizer.minimize(loss)
=======
#!/usr/bin/env python3
"""
Defines function that creates the training op
for a neural network in TensorFlow using
the gradient descent with momentum optimization algorithm
"""


import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    Creates the training operation for a neural network in TensorFlow
        using the gradient descent with momentum optimization algorithm

    parameters:
        loss: the loss of the network
        alpha [float]: learning rate
        beta1 [float]: momentum weight

    returns:
        the momentum optimization operation
    """
    op = tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
    return op
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
