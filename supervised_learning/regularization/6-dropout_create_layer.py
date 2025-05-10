<<<<<<< HEAD
#!/usr/bin/env python3
""" Create a Layer with Dropout"""


import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """ Create a Layer with Dropout

    Args:
        prev (_type_): _description_
        n (_type_): _description_
        activation (_type_): _description_
        keep_prob (_type_): _description_
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init)
    dropout = tf.layers.Dropout(rate=keep_prob)
    return dropout(layer(prev))
=======
#!/usr/bin/env python3
'''
This script creates a dropout layer
using tensorflow library.
'''

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    '''
    The function dropout_create_layer creates a dropout layer.
    It uses the tensorflow library.

    Arguments:
    prev -- tensor output of the previous layer
    n -- number of nodes in the layer to create
    activation -- activation function that should be used on the layer
    keep_prob -- probability that a node will be kept
    '''
    kernel_ini = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    kernel_reg = tf.layers.Dropout(keep_prob)
    layer = tf.layers.Dense(name='layer', units=n, activation=activation,
                            kernel_initializer=kernel_ini,
                            kernel_regularizer=kernel_reg)

    return layer(prev)
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
