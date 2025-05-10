#!/usr/bin/env python3
<<<<<<< HEAD
""" Neural Network
"""
=======
"""
File defines a class that represents a
neural network with one hidden layer
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
    Class that represents a neural network with one hidden layer
    """

    def __init__(self, nx, nodes):
        """
        Initializes a neural network with one hidden layer

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
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

<<<<<<< HEAD
    # getter functions
    @property
    def W1(self):
        """Return weights vector for hidden layer"""
        return self.__W1

    @property
    def b1(self):
        """Return bias for hidden layer"""
        return self.__b1

    @property
    def A1(self):
        """Return activated output for hidden layer"""
        return self.__A1

    @property
    def W2(self):
        """Return weights vector for output neuron"""
        return self.__W2

    @property
    def b2(self):
        """Return bias for the output neuron"""
        return self.__b2

    @property
    def A2(self):
        """Return activated output for the output neuron"""
        return self.__A2

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neural network

        Args:
            X (numpy.array): Input data with shape (nx, m)
        """
        z = np.matmul(self.__W1, X) + self.__b1
        sigmoid = 1 / (1 + np.exp(-z))
        self.__A1 = sigmoid
        z = np.matmul(self.__W2, self.__A1) + self.__b2
        sigmoid = 1 / (1 + np.exp(-z))
        self.__A2 = sigmoid
        return self.__A1, self.__A2
=======
    @property
    def W1(self):
        """
        Getter for the weights vector for the hidden layer
        """
        return (self.__W1)

    @property
    def b1(self):
        """
        Getter for the bias for the hidden layer
        """
        return (self.__b1)

    @property
    def A1(self):
        """
        Getter for the activated output for the hidden layer
        """
        return (self.__A1)

    @property
    def W2(self):
        """
        Getter for the weights vector for the output neuron
        """
        return (self.__W2)

    @property
    def b2(self):
        """
        Getter for the bias for the output neuron
        """
        return (self.__b2)

    @property
    def A2(self):
        """
        Getter for the activated output for the output neuron
        """
        return (self.__A2)

    def forward_prop(self, X):
        """
        Computes forward propagation for the neural network
        """
        z1 = np.matmul(self.W1, X) + self.b1
        self.__A1 = 1 / (1 + (np.exp(-z1)))
        z2 = np.matmul(self.W2, self.__A1) + self.b2
        self.__A2 = 1 / (1 + (np.exp(-z2)))
        return (self.A1, self.A2)
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
