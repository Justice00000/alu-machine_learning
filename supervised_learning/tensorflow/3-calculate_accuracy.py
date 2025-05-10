#!/usr/bin/env python3
<<<<<<< HEAD
""" create layer """
=======
"""
Defines a function that calculates the accuracy of a prediction
for the neural network
"""


>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
import tensorflow as tf


def calculate_accuracy(y, y_pred):
<<<<<<< HEAD
    """ calculate accuracy"""
    p_m = tf.arg_max(y_pred, 1)
    y_m = tf.arg_max(y, 1)
    e = tf.equal(y_m, p_m)
    return tf.reduce_mean(tf.cast(e, tf.float32))
=======
    """
    Calculates the accuracy of a prediction for the neural network

    parameters:
        y [tf.placeholder]: placeholder for labels of the input data
        y_pred [tensor]: contains network's predictions

    returns:
        tensor containing decimal accuracy of the prediction
    """
    y_pred = tf.math.argmax(y_pred, axis=1)
    y = tf.math.argmax(y, axis=1)
    equality = tf.math.equal(y_pred, y)
    accuracy = tf.reduce_mean(tf.cast(equality, "float"))
    return accuracy
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
