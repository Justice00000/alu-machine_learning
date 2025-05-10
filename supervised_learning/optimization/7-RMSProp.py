<<<<<<< HEAD
#!/usr/bin/env python3
"""
    function def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    that updates a variable using the RMSProp optimization algorithm:
"""


import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Args:
        - alpha is the learning rate
        - beta2 is the RMSProp weight
        - epsilon is a small number to avoid division by zero
        - var is a numpy.ndarray containing the variable to be updated
        - grad is a numpy.ndarray containing the gradient of var
        - s is the previous second moment of var

    Returns:
        The updated variable and the new moment, respectively
    """
    s = beta2 * s + (1 - beta2) * np.power(grad, 2)
    var = var - alpha * grad / (np.sqrt(s) + epsilon)
    return var, s
=======
#!/usr/bin/env python3
"""
Defines function that updates a variable
using RMSProp optimization algorithm
"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using RMSProp optimization algorithm

    parameters:
        alpha [float]: learning rate
        beta2 [float]: RMSProp weight
        epsilon [float]:
            small number to avoid division by zero
        var [numpy.ndarray]:
            contains the variance to be updated
        grad [numpy.ndarray]:
            contains the gradient of var
        s [tf.moment]:
            the previous second moment of var

    s_dW = (beta * s_dW) + ((1 - beta) * (dW ** 2))
    W = W - (alpha * (dW / sqrt(s_dW)))

    returns:
        the updated variable and the new moment, respectively
    """
    s_dW = (beta2 * s) + ((1 - beta2) * (grad ** 2))
    var -= alpha * (grad / (epsilon + (s_dW ** (1 / 2))))
    return var, s_dW
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
