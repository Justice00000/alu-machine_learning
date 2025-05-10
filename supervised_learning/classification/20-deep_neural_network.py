#!/usr/bin/env python3
<<<<<<< HEAD
""" Deep Neural Network
"""
=======
"""
defines DeepNeuralNetwork class that defines
a deep neural network performing binary classification
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

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError('layers must be a list of positive integers')

            if i == 0:
                # He et al. initialization
                self.__weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], nx) * np.sqrt(2 / nx)
            else:
                # He et al. initialization
                self.__weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])

            # Zero initialization
            self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    # add getter method
    @property
    def L(self):
        """ Return layers in the neural network"""
        return self.__L

    @property
    def cache(self):
        """ Return dictionary with intermediate values of the network"""
        return self.__cache

    @property
    def weights(self):
        """Return weights and bias dictionary"""
        return self.__weights

    def forward_prop(self, X):
        """ Forward propagation

        Args:
            X (numpy.array): Input array with
            shape (nx, m) = (featurs, no of examples)
        """
        self.cache["A0"] = X
        # print(self.cache)
        for i in range(1, self.L+1):
            # extract values
            W = self.weights['W'+str(i)]
            b = self.weights['b'+str(i)]
            A = self.cache['A'+str(i - 1)]
            # do forward propagation
            z = np.matmul(W, A) + b
            sigmoid = 1 / (1 + np.exp(-z))  # this is the output
            # store output to the cache
            self.cache["A"+str(i)] = sigmoid
        return self.cache["A"+str(i)], self.cache

    def cost(self, Y, A):
        """ Calculate the cost of the Neural Network.

        Args:
            Y (numpy.array): Actual values
            A (numpy.array): predicted values of the neural network

        Returns:
            _type_: _description_
        """
        loss = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost = np.mean(loss)
        return cost

    def evaluate(self, X, Y):
        """ Evaluate the neural network

        Args:
            X (numpy.array): Input array
            Y (numpy.array): Actual values

        Returns:
            prediction, cost: return predictions and costs
        """
        self.forward_prop(X)
        # get output of the neural network from the cache
        output = self.cache.get("A" + str(self.L))
        return np.where(output >= 0.5, 1, 0), self.cost(Y, output)
=======
    """
    class that represents a deep neural network
    performing binary classification

    class constructor:
        def __init__(self, nx, layers)

    private instance attributes:
        L: the number of layers in the neural network
        cache: a dictionary holding all intermediary values of the network
        weights: a dictionary holding all weights and biases of the network

    public methods:
        def forward_prop(self, X):
            calculates the forward propagation of the neural network
        def cost(self, Y, A):
            calculates the cost of the model using logistic regression
        def evaluate(self, X, Y):
            evaluates the neural network's predictions
    """

    def __init__(self, nx, layers):
        """
        class constructor

        parameters:
            nx [int]: the number of input features
                If nx is not an integer, raise a TypeError.
                If nx is less than 1, raise a ValueError.
            layers [list]: representing the number of nodes in each layer
                If layers is not a list, raise TypeError.
                If elements in layers are not all positive ints,
                    raise a TypeError.

        sets private instance attributes:
            __L: the number of layers in the neural network,
                initialized based on layers
            __cache: a dictionary holding all intermediary values for network,,
                initialized as an empty dictionary
            __weights: a dictionary holding all weights/biases of the network,
                weights initialized using the He et al. method
                    using the key W{l} where {l} is the hidden layer
                biases initialized to 0s
                    using the key b{l} where {1} is the hidden layer
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
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = weights

    @property
    def L(self):
        """
        gets the private instance attribute __L
        __L is the number of layers in the neural network
        """
        return (self.__L)

    @property
    def cache(self):
        """
        gets the private instance attribute __cache
        __cache holds all the intermediary values of the network
        """
        return (self.__cache)

    @property
    def weights(self):
        """
        gets the private instance attribute __weights
        __weights holds all the wrights and biases of the network
        """
        return (self.__weights)

    def forward_prop(self, X):
        """
        calculates the forward propagation of the neuron

        parameters:
            X [numpy.ndarray with shape (nx, m)]: contains the input data
                nx is the number of input features to the neuron
                m is the number of examples

        updates the private attribute __cache using sigmoid activation function
        sigmoid function:
            activated output = 1 / (1 + e^(-z))
            z = sum of ((__Wi * __Xi) + __b) from i = 0 to nx
        activated outputs of each layer are saved in __cache
            as A{l} where {l} is the hidden layer
        X is saved to __cache under key A0

        return:
            the output of the neural network and the cache, respectively
        """
        self.__cache["A0"] = X
        for index in range(self.L):
            W = self.weights["W{}".format(index + 1)]
            b = self.weights["b{}".format(index + 1)]
            z = np.matmul(W, self.cache["A{}".format(index)]) + b
            A = 1 / (1 + (np.exp(-z)))
            self.__cache["A{}".format(index + 1)] = A
        return (A, self.cache)

    def cost(self, Y, A):
        """
        calculates the cost of the model using logistic regression

        parameters:
            Y [numpy.ndarray with shape (1, m)]:
                contains correct labels for the input data
            A [numpy.ndarray with shape (1, m)]:
                contains the activated output of the neuron for each example

        logistic regression loss function:
            loss = -((Y * log(A)) + ((1 - Y) * log(1 - A)))
            To avoid log(0) errors, uses (1.0000001 - A) instead of (1 - A)
        logistic regression cost function:
            cost = (1 / m) * sum of loss function for all m example

        return:
            the calculated cost
        """
        m = Y.shape[1]
        m_loss = np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A)))
        cost = (1 / m) * (-(m_loss))
        return (cost)

    def evaluate(self, X, Y):
        """
        evaluates the neural network's predictions

        parameters:
            X [numpy.ndarray with shape (nx, m)]: contains the input data
                nx is the number of input features to the neuron
                m is the number of examples
            Y [numpy.ndarray with shape (1, m)]:
                contains correct labels for the input data

        returns:
            the neuron's prediction and the cost of the network, respectively
            prediction is numpy.ndarray with shape (1, m), containing
                predicted labels for each example
            label values should be 1 if the output of the network is >= 0.5,
                0 if the output of the network is < 0.5
        """
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return (prediction, cost)
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
