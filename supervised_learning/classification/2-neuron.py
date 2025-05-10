#!/usr/bin/env python3
<<<<<<< HEAD
"""Class Neuron that defines a single neuron performing binary classification
=======
"""
defines Neuron class that defines
a single neuron performing binary classification
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
"""


import numpy as np


class Neuron:
<<<<<<< HEAD
    """ Class Neuron
    """

    def __init__(self, nx):
        """ Instantiation function of the neuron

        Args:
            nx (_type_): _description_

        Raises:
            TypeError: _description_
            ValueError: _description_
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be positive')

        # initialize private instance attributes
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

        # getter function
    @property
    def W(self):
        """Return weights"""
        return self.__W

    @property
    def b(self):
        """Return bias"""
        return self.__b

    @property
    def A(self):
        """Return output"""
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron

        Args:
            X (numpy.ndarray): matrix with the input data of shape (nx, m)

        Returns:
            numpy.ndarray: The output of the neural network.
        """
        z = np.matmul(self.__W, X) + self.__b
        sigmoid = 1 / (1 + np.exp(-z))
        self.__A = sigmoid
        return self.__A
=======
    """
    Defines a single neuron to mimic binary classification

    class constructor
    """

    def __init__(self, nx):
        """
        Initializes a neuron given the number of input features

        this is usually how we define the input features in a neuron
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        Getter for the private instance attribute __W
        """
        return (self.__W)

    @property
    def b(self):
        """
        Getter for the private instance attribute __b
        """
        return (self.__b)

    @property
    def A(self):
        """
        Getter for the private instance attribute __A
        """
        return (self.__A)

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron

        This serves as the output of the neuron
        """
        z = np.matmul(self.W, X) + self.b
        self.__A = 1 / (1 + (np.exp(-z)))
        return (self.A)
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
