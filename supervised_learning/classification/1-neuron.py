#!/usr/bin/env python3
<<<<<<< HEAD
"""Class Neuron that defines a single neuron performing binary classification
=======
"""
The definition of a single neuron performing binary classification

A bit more complex than the previous version
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
            raise ValueError('nx must be a positive')

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
=======
    """
    This class mimics the behavior of a single neuron in a neural network

    nx: The number of input features to the neuron
    """

    def __init__(self, nx):
        """
        Initializes a neuron

        nx is a positive integer that represents the number
        of input features to the neuron
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
        Getter for private class attribute __W
        """
        return (self.__W)

    @property
    def b(self):
        """
        Getter for private class attribute __b
        """
        return (self.__b)

    @property
    def A(self):
        """
        Getter for private class attribute __A
        """
        return (self.__A)
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
