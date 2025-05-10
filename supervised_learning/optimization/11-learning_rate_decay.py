<<<<<<< HEAD
#!/usr/bin/env python3
""" Learing rate decay
"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ updates the learning rate using inverse time decay in numpy

    Args:
        alpha (float): original learning rate
        decay_rate (float): weight used to determine the rate at
        which alpha will decay
        global_step (int): number of passes of gradient descent
        that have elapsed
        decay_step (int): number of passes of gradient descent that
        should occur before alpha is decayed further
    """
    return alpha / (1 + decay_rate * (global_step // decay_step))
=======
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 18:12:52 2021

@author: Robinson Montes
"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Function that updates the learning rate using inverse time decay in numpy
    Arguments:
     - alpha is the original learning rate
     - decay_rate is the weight used to determine the rate at which alpha will
        decay
     - global_step is the number of passes of gradient descent that
        have elapsed
     - decay_step is the number of passes of gradient descent that
        should occur before alpha is decayed further
    Returns:
     The updated value for alpha
    """

    decay = alpha / (1 + decay_rate * np.floor(global_step / decay_step))

    return decay
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
