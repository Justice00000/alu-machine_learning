#!/usr/bin/env python3
<<<<<<< HEAD
"""Class Neuron that defines a single neuron performing binary classification
=======
"""
Defines a function to return two placeholders for the neural network
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
"""


import tensorflow as tf


def create_placeholders(nx, classes):
<<<<<<< HEAD
    """Function that returns two placeholders, x and y, for the neural network

    Args:
        nx (_type_): _description_
        classes (_type_): _description_
    """
    x = tf.placeholder("float", shape=[None, nx], name="x")
    y = tf.placeholder("float", shape=[None, classes], name="y")
=======
    """
    Returns two placeholders, x and y, for the neural network
    x is the placeholder for input data to the neural network
    y is the placeholder for the one-hot labels for the input data

    parameters:
        nx [int]: the number of feature columns in the data
        classes [int]: the number of classes in the classifier

    returns:
        the placeholders, x and y, respectively
    """
    x = tf.placeholder("float", shape=(None, nx), name="x")
    y = tf.placeholder("float", shape=(None, classes), name="y")
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
    return x, y
