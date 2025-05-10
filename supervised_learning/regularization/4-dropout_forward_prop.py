<<<<<<< HEAD
#!/usr/bin/env python3
""" Forward Propagation with Dropout """


import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """ Forward Propagation with Dropout
        X: (nx, m) input data
          nx: number of input features
          m: number of examples
        weights: dictionary of weights and biases of the neural network
        L: number of layers in the network
        keep_prob: probability that a node will be kept
        Returns: dictionary containing the outputs of each
        layer and the dropout mask used on each layer
    """
    cache = {}
    cache['A0'] = X
    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        A = cache['A' + str(i - 1)]
        Z = np.matmul(W, A) + b
        if i < L:
            A = np.tanh(Z)
            D = np.random.rand(A.shape[0], A.shape[1])
            D = np.where(D < keep_prob, 1, 0)
            A = np.multiply(A, D)
            A = A / keep_prob
            cache['D' + str(i)] = D
        else:
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
        cache['A' + str(i)] = A
    return cache
=======
#!/usr/bin/env python3
'''
This script implements the forward propagation of the dropout
layer. The dropout layer is a regularization technique that
prevents overfitting by randomly setting some neurons to zero
during training. This script implements the forward propagation
of the dropout layer.
'''

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    '''
    The goal of this function is to implement the forward
    propagation of the dropout layer. The dropout layer is
    a regularization technique that prevents overfitting by
    randomly setting some neurons to zero during training.

    Arguments:
        X : numpy array - input data
        weights : dict - weights of the neural network
        L : int - number of layers in the neural network
        keep_prob : float - probability of keeping a neuron active
    '''
    outputs = {}
    outputs["A0"] = X
    for index in range(L):
        weight = weights["W{}".format(index + 1)]
        bias = weights["b{}".format(index + 1)]
        z = np.matmul(weight, outputs["A{}".format(index)]) + bias
        dropout = np.random.binomial(1, keep_prob, size=z.shape)
        if index != (L - 1):
            A = np.tanh(z)
            A *= dropout
            A /= keep_prob
            outputs["D{}".format(index + 1)] = dropout
        else:
            A = np.exp(z)
            A /= np.sum(A, axis=0, keepdims=True)
        outputs["A{}".format(index + 1)] = A
    return outputs
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
