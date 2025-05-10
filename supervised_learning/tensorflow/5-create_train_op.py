#!/usr/bin/env python3
<<<<<<< HEAD
""" train"""
=======
"""
Defines a function that creates the training operation
for the neural network
"""


>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
import tensorflow as tf


def create_train_op(loss, alpha):
<<<<<<< HEAD
    """ training operation"""
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
=======
    """
    Creates the training operation for the network

    parameters:
        loss [tensor]: loss of the network's prediction
        alpha [float]: learning rate

    returns:
        operation that trains the network using gradient descent
    """
    gradient_descent = tf.train.GradientDescentOptimizer(alpha)
    return (gradient_descent.minimize(loss))
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
