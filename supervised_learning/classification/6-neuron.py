#!/usr/bin/env python3
<<<<<<< HEAD
"""Class Neuron that defines a single neuron performing binary classification
=======
"""
Defines Neuron class that defines
a singular neuron performing binary classification
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
            nx (int): number of features to be initialized

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

    def cost(self, Y, A):
        """ Compute the of the model using logistic regression

        Args:
            Y (np.array): True values
            A (np.array): Prediction valuesss

        Returns:
            float: cost function
        """
        # calculate
        loss = - (Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost = np.mean(loss)
        return cost

    def evaluate(self, X, Y):
        """ Evaluate the cost function

        Args:
            X (_type_): _description_
            Y (_type_): _description_

        Returns:
            _type_: _description_
        """
        pred = self.forward_prop(X)
        cost = self.cost(Y, pred)
        pred = np.where(pred > 0.5, 1, 0)
        return (pred, cost)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ Calculate one pass of gradient descent on the neuron

        Args:
            X (_type_): _description_
            Y (_type_): _description_
            A (_type_): _description_
            alpha (float, optional): _description_. Defaults to 0.05.
        """
        dz = A - Y
        m = X.shape[1]
        dw = (1/m) * np.matmul(dz, X.T)
        db = np.mean(dz)
        self.__W -= alpha * dw
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ Train the neuron: finding the global minuminus of the cost function

        Args:
            X (np.array): _description_
            Y (np.array): _description_
            iterations (int, optional): _description_. Defaults to 5000.
            alpha (float, optional): _description_. Defaults to 0.05.
        """
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')

        for _ in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
        return self.evaluate(X, Y)
=======
    """
    Class that defines a single neuron
    performing binary classification
    """

    def __init__(self, nx):
        """
        Initializes a single neuron
        performing binary classification
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
        Getter for the weight attribute
        """
        return (self.__W)

    @property
    def b(self):
        """
        Getter for the bias attribute
        """
        return (self.__b)

    @property
    def A(self):
        """
        Getter for the activation attribute
        """
        return (self.__A)

    def forward_prop(self, X):
        """
        Computes the forward propagation of the neuron
        """
        z = np.matmul(self.W, X) + self.b
        self.__A = 1 / (1 + (np.exp(-z)))
        return (self.A)

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        """
        m = Y.shape[1]
        m_loss = np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A)))
        cost = (1 / m) * (-(m_loss))
        return (cost)

    def evaluate(self, X, Y):
        """
        Computes the prediction and cost of the network
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return (prediction, cost)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Computes one pass of gradient descent on the neuron
        """
        m = Y.shape[1]
        dz = (A - Y)
        d__W = (1 / m) * (np.matmul(X, dz.transpose()).transpose())
        d__b = (1 / m) * (np.sum(dz))
        self.__W = self.W - (alpha * d__W)
        self.__b = self.b - (alpha * d__b)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Training function for the neuron

        - X is the numpy.ndarray with shape (nx, m) that
        contains the input data
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for itr in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
        return (self.evaluate(X, Y))
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
