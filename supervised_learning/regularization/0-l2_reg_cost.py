<<<<<<< HEAD
#!/usr/bin/env python3
"""
 Calculates the cost of a neural network with L2 regularization
"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ Calculates the cost of a neural network with L2 regularization

    Args:
        cost (float): the cost of the network without L2 regularization
        lambtha (_type_):  is the regularization parameter
        weights (_type_): s a dictionary of the weights and biases
        (numpy.ndarrays) of the neural network
        L (int): the number of layers in the neural network
        m (int): the number of data points used
    """
    penality = 0

    for i in range(1, L + 1):
        key = 'W' + str(i)
        penality += np.sum(np.square(weights[key]))

    penality *= (lambtha / (2 * m))

    total_cost = cost + penality

    return total_cost
=======
#!/usr/bin/env python3
'''
File to calculate the cost of a neural network with L2
regularization. The cost is calculated using the formula:

C = J + (λ / (2 * m)) * Σ||W||^2
'''

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    '''
    Function that calculates the cost of a neural
    network with L2 regularization

    Parameters:
    cost: float - cost of the network without L2 regularization
    lambtha: float - regularization parameter
    weights: dict - weights and biases of the network
    L: int - number of layers in the network
    m: int - number of data points used
    '''
    weights_squared_sum = 0

    for i in range(L):
        W = weights['W' + str(i+1)]
        weights_squared_sum += np.sum(np.square(W))

    l2_cost = (lambtha / (2 * m)) * weights_squared_sum
    return np.array([float(cost + l2_cost)])
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
