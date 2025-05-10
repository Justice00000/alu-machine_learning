#!/usr/bin/env python3
<<<<<<< HEAD
"""Class Neuron that defines a single neuron performing binary classification
"""

=======
"""
Defines a single neuron performing binary classification. Neural network
"""
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6

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
            raise ValueError('nx must be a positive integer')

        self.W = np.random.normal(size=(1, nx))
=======
    """
    This class defines a single neuron that mimicas the behaavior of tensorflow

    By default, the bias b is initialized to 0.
    Upon instantiation, a neuron takes in a single parameter:
    """

    def __init__(self, nx):
        """
        class constructor

        This is how the class is called upon instatiation
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.randn(1, nx)
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
        self.b = 0
        self.A = 0
