#!/usr/bin/env python3
<<<<<<< HEAD
""" Deep Neural Network
"""
=======
"""
File defines a class that represents a deep neural network
"""

>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6

import numpy as np


class DeepNeuralNetwork:
<<<<<<< HEAD
    """ Class that defines a deep neural network performing binary
        classification.
    """

    def __init__(self, nx, layers):
        """ Instantiation function

        Args:
            nx (int): number of input features
            layers (list): representing the number of nodes in each layer of
                           the network
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')

        if not isinstance(layers, list):
            raise TypeError('layers must be a list of positive integers')
        if len(layers) < 1:
            raise TypeError('layers must be a list of positive integers')

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i in range(self.L):
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError('layers must be a list of positive integers')

            if i == 0:
                # He et al. initialization
                self.weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], nx) * np.sqrt(2 / nx)
            else:
                # He et al. initialization
                self.weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])

            # Zero initialization
            self.weights['b' + str(i + 1)] = np.zeros((layers[i], 1))
=======
    """
    Class that represents a deep neural network
    """

    def __init__(self, nx, layers):
        """
        Initializes a deep neural network
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")
        weights = {}
        previous = nx
        for index, layer in enumerate(layers, 1):
            if type(layer) is not int or layer < 0:
                raise TypeError("layers must be a list of positive integers")
            weights["b{}".format(index)] = np.zeros((layer, 1))
            weights["W{}".format(index)] = (
                np.random.randn(layer, previous) * np.sqrt(2 / previous))
            previous = layer
        self.L = len(layers)
        self.cache = {}
        self.weights = weights
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
