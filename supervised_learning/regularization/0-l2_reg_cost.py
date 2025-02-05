#!/usr/bin/env python3

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
