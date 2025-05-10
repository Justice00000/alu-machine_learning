<<<<<<< HEAD
#!/usr/bin/env python3
"""
Compute gradient descent with L2 regularization
"""


import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ Compute gradient descent with L2 regularization

    Args:
        Y (numpy.ndarray): one-hot matrix with the correct labels
        weights (dict): The weights and biases of the network
        cache (dict): The outputs of each layer of the network
        alpha (float): The learning rate
        lambtha (float): The L2 regularization parameter
        L (int): The number of layers of the network
    """
    m = Y.shape[1]
    dz = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        A = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        dw = (1 / m) * np.matmul(dz, A.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        dz = np.matmul(W.T, dz) * (1 - np.square(A))
        weights['W' + str(i)] = weights['W' + str(i)] - alpha * dw
        weights['b' + str(i)] = weights['b' + str(i)] - alpha * db

    return weights
=======
#!/usr/bin/env python3
'''
Function that updates the weights and biases of a neural network using
gradient descent with L2 regularization.
'''

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates weights and biases using gradient descent with L2 regularization
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for l in range(L, 0, -1):
        A_prev = cache['A' + str(l-1)]

        # Calculate gradients with L2 regularization
        dW = (1/m) * np.matmul(dZ, A_prev.T) + \
            (lambtha/m) * weights['W' + str(l)]
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)

        # Calculate dZ for next layer if not at first layer
        if l > 1:
            W = weights['W' + str(l)]
            dZ = np.matmul(W.T, dZ) * (1 - np.square(cache['A' + str(l-1)]))

        # Update weights and biases
        weights['W' + str(l)] = weights['W' + str(l)] - alpha * dW
        weights['b' + str(l)] = weights['b' + str(l)] - alpha * db
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
