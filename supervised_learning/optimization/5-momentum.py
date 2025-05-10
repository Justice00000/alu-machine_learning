<<<<<<< HEAD
#!/usr/bin/env python3
""" Momentum
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """ Momentum

    Args:
        alpha (float): learning rate
        beta1 (float): momentum weight
        var (np.ndarray): variable to be updated
        grad (np.ndarray): gradient of var
        v (np.ndarray): the previous first moment of var
    Returns:
        np.ndarray: the updated variable and the new moment, respectively
    """
    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v
    return var, v
=======
#!/usr/bin/env python3
"""
Defines function that updates a variable
using gradient descent with momentum optimization algorithm
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using gradient descent
        with momentum optimization algorithm

    parameters:
        alpha [float]: learning rate
        beta1 [float]: momentum weight
        var [numpy.ndarray]:
            contains the variance to be updated
        grad [numpy.ndarray]:
            contains the gradient of var
        v [tf.moment]:
            the previous first moment of var

    v_dW = (beta * v_dW) + ((1 - beta) * dW)
    W = W - (alpha * v_dW)

    returns:
        the updated variable and the new moment, respectively
    """
    dW_prev = (beta1 * v) + ((1 - beta1) * grad)
    var -= (alpha * dW_prev)
    return var, dW_prev
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
