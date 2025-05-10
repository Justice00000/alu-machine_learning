<<<<<<< HEAD
#!/usr/bin/env python3
""" Learing rate decay with tensorflow
"""


import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ creates a learning rate decay operation in tensorflow using
        inverse time decay

    Args:
        alpha (float): original learning rate
        decay_rate (float): weight used to determine the rate at
        which alpha will decay
        global_step (int): number of passes of gradient descent
        that have elapsed
        decay_step (int): number of passes of gradient descent that
        should occur before alpha is decayed further
    """
    return tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                       decay_rate, staircase=True)
=======
#!/usr/bin/env python3
"""
Defines function that creates a learning rate decay op
for a neural network in TensorFlow using inverse time decay
"""


import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Creates a learning rate decay op for a neural network in TensorFlow
        using inverse time decay

    parameters:
        alpha [float]: original learning rate
        decay_rate: wight used to determine the rate at which alpha will decay
        global_step [int]:
            number of passes of gradient descent that have elapsed
        decay_step [int]:
            number of passes of gradient descent that should occur before
                alpha is decayed furtherXS

    the learning rate decay should occur in a stepwise fashion

    returns:
        the learning rate decay operation
    """
    op = tf.train.inverse_time_decay(
        alpha, global_step, decay_step, decay_rate, staircase=True)
    return op
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
