<<<<<<< HEAD
#!/usr/bin/env python3
""" Batch Normalization with tensorflow
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """ Batch Normalization with tensorflow

    Args:
        prev (): is the activated output of the previous layer
        n (int): is the number of nodes in the layer to be created
        activation (): is the activation function that should be used on the
        output of the layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    model = tf.layers.Dense(units=n, kernel_initializer=init)
    Z = model(prev)
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), name='gamma')
    beta = tf.Variable(tf.constant(0.0, shape=[n]), name='beta')
    mean, variance = tf.nn.moments(Z, axes=[0])
    epsilon = 1e-8
    Z_norm = tf.nn.batch_normalization(Z, mean, variance, beta, gamma, epsilon)
    return activation(Z_norm)
=======
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 18:12:52 2021

@author: Robinson Montes
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Function that creates a batch normalization layer for a NN in tensorflow:
    Arguments:
     - prev is the activated output of the previous layer
     - n is the number of nodes in the layer to be created
     - activation is the activation function that should be used on
        the output of the layer
    Returns:
     A tensor of the activated output for the layer
    """

    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, kernel_initializer=init)

    z = layer(prev)

    mt, vt = tf.nn.moments(z, [0])
    beta = tf.Variable(tf.zeros([z.get_shape()[-1]]))
    gamma = tf.Variable(tf.ones([z.get_shape()[-1]]))
    zt = tf.nn.batch_normalization(z, mt, vt, beta, gamma, 1e-8)
    y_pred = activation(zt)

    return y_pred
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
