<<<<<<< HEAD
#!/usr/bin/env python3
""" L2 Regularization Cost
"""


import tensorflow as tf


def l2_reg_cost(cost):
    """ Calculates the cost of a neural network with L2 regularization

    Args:
        cost (float): is a tensor containing the cost of the network
        without L2 regularization
    """
    return cost + tf.losses.get_regularization_losses()
=======
#!/usr/bin/env python3
'''
Implement the L2 regularization cost function
'''
import tensorflow as tf


def l2_reg_cost(cost):
    '''
    Calculate the L2 regularization cost

    Arguments:
    cost -- cost of the neural network without regularization
    '''
    l2_reg_cost = tf.losses.get_regularization_losses()
    return (cost + l2_reg_cost)
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
