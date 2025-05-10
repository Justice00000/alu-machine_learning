#!/usr/bin/env python3
<<<<<<< HEAD
""" Neural Network
"""
=======
"""
Defines a neural network with one hidden layer
"""

>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6

import numpy as np


class NeuralNetwork:
<<<<<<< HEAD
    """ Class that defines a neural network with one hidden layer performing
        binary classification.
    """

    def __init__(self, nx, nodes):
        """ Instantiation function

        Args:
            nx (int): size of the input layer
            nodes (_type_): _description_
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')

        if not isinstance(nodes, int):
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

=======
    """
    Definition of NeuralNetwork class containing one hidden layer
    """

    def __init__(self, nx, nodes):
        """
        Initializes the NeuralNetwork instance
        nx is the number of input features to the neuron
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
