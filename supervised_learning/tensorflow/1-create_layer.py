#!/usr/bin/env python3
<<<<<<< HEAD
""" create layer """
=======
"""
Defines a function to create a layer for neural network
"""


>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
import tensorflow as tf


def create_layer(prev, n, activation):
<<<<<<< HEAD
    """ create layer"""
    W = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    L = tf.layers.Dense(units=n, activation=activation,
                        kernel_initializer=W, name="layer")
    return L(prev)
=======
    """
    Creates a layer for neural network

    parameters:
        prev [tensor]: tensor output of the previous layer
        n [int]: the number of nodes in the layer to create
        activation [function]: the activation function the layer should use

    use tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
        to implement He et. al initialization for the layer weights
    each layer is given the name "layer"

    returns:
        tensor output of the layer
    """
    weights_initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(
        n,
        activation=activation,
        name="layer",
        kernel_initializer=weights_initializer)
    return (layer(prev))
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
