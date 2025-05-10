#!/usr/bin/env python3
<<<<<<< HEAD
""" Deep Neural Network
"""

=======
# -*- coding: utf-8 -*-
"""
Update the class DeepNeuralNetwork to allow
different activation functions
"""
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
<<<<<<< HEAD
    """ Class that defines a deep neural network performing binary
        classification.
    """

    def __init__(self, nx, layers, activation='sig'):
        """ Instantiation function

        Args:
            nx (int): number of input features
            layers (list): representing the number of nodes in each layer of
                           the network
            activation (str, optional): Activation function for hidden layers.
                                        Defaults to 'sig'.
        """
        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")
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
        self.__activation = activation

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

    # getter methods

    @property
    def activation(self):
        """ Return activation function """
        return self.__activation

    @property
    def L(self):
        """ Return layers in the neural network"""
=======
    """
    Class DeepNeuralNetwork that defines a deep neural network
    performing binary classification
    """

    def __init__(self, nx, layers, activation='sig'):
        """
        Constructor for the class
        Arguments:
         - nx (int): is the number of input features to the neuron
         - layers (list): representing the number of nodes in each layer of
                          the network
        Public instance attributes:
         - L: The number of layers in the neural network.
         - cache: A dictionary to hold all intermediary values of the network.
         - weights: A dictionary to hold all weights and biased of the network.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(layers) != list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__activation = activation
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.L):
            if type(layers[i]) is not int or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            wkey = "W{}".format(i + 1)
            bkey = "b{}".format(i + 1)

            self.__weights[bkey] = np.zeros((layers[i], 1))

            if i == 0:
                w = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
            else:
                w = np.random.randn(layers[i], layers[i - 1])
                w = w * np.sqrt(2 / layers[i - 1])
            self.__weights[wkey] = w

    @property
    def L(self):
        """
        getter function for L
        Returns the number of layers
        """
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
        return self.__L

    @property
    def cache(self):
<<<<<<< HEAD
        """ Return dictionary with intermediate values of the network"""
=======
        """
        getter gunction for cache
        Returns a dictionary to hold all intermediary values of the network
        """
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
        return self.__cache

    @property
    def weights(self):
<<<<<<< HEAD
        """Return weights and bias dictionary"""
        return self.__weights

    def forward_prop(self, X):
        """ Forward propagation """
        self.cache["A0"] = X
        for i in range(1, self.L+1):
            W = self.weights['W'+str(i)]
            b = self.weights['b'+str(i)]
            A = self.cache['A'+str(i - 1)]
            z = np.matmul(W, A) + b
            if i != self.L:
                if self.activation == 'sig':
                    A = 1 / (1 + np.exp(-z))  # sigmoid function
                elif self.activation == 'tanh':
                    A = np.tanh(z)  # tanh function
            else:
                A = np.exp(z) / np.sum(np.exp(z), axis=0)  # softmax function
            self.cache["A"+str(i)] = A
        return self.cache["A"+str(i)], self.cache

    def cost(self, Y, A):
        """ Calculate the cost of the Neural Network \
            using categorical cross-entropy.

        Args:
            Y (numpy.array): Actual one-hot encoded \
                labels with shape (classes, m)
            A (numpy.array): Predicted probabilities \
                from the output layer of the neural network

        Returns:
            float: Categorical cross-entropy cost
        """
        cost = -np.sum(Y * np.log(A)) / Y.shape[1]
        return cost

    def evaluate(self, X, Y):
        """ Evaluate the neural network

        Args:
            X (numpy.array): Input array
            Y (numpy.array): Actual one-hot encoded labels

        Returns:
            prediction, cost: return predictions and costs
        """
        self.forward_prop(X)
        # get output of the neural network from the cache
        A = self.cache.get("A" + str(self.L))
        # get the class with the highest probability
        prediction = np.eye(A.shape[0])[np.argmax(A, axis=0)].T
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Calculate one pass of gradient descent on the neural network

        Args:
            Y (numpy.array): Actual one-hot encoded labels
            cache (dict): Dictionary containing all intermediary values of the
                        network
            alpha (float): learning rate
        """
        m = Y.shape[1]

        for i in range(self.L, 0, -1):

            A_prev = cache["A" + str(i - 1)]
            A = cache["A" + str(i)]
            W = self.weights["W" + str(i)]

            if i == self.L:
                dz = A - Y
            else:
                if self.activation == 'sig':
                    dz = da * (A * (1 - A))  # sigmoid derivative
                elif self.activation == 'tanh':
                    dz = da * (1 - A**2)  # tanh derivative

            db = dz.mean(axis=1, keepdims=True)
            dw = np.matmul(dz, A_prev.T) / m
            da = np.matmul(W.T, dz)
            self.weights['W' + str(i)] -= (alpha * dw)
            self.weights['b' + str(i)] -= (alpha * db)

    def train(self, X, Y, iterations=5000,
              alpha=0.05, verbose=True, graph=True, step=100):
        """ Train the deep neural network

        Args:
            X (_type_): _description_
            Y (_type_): _description_
            iterations (int, optional): _description_. Defaults to 5000.
            alpha (float, optional): _description_. Defaults to 0.05.
            verbose (bool, optional): _description_. Defaults to True.
            graph (bool, optional): _description_. Defaults to True.
            step (int, optional): _description_. Defaults to 100.

        Raises:
            TypeError: _description_
            ValueError: _description_
            TypeError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """

        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations < 1:
            raise ValueError('iterations must be a positive integer')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')

        costs = []
        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(Y, self.cache, alpha)
            if verbose and i % step == 0:

                cost = self.cost(Y, self.cache["A"+str(self.L)])
                costs.append(cost)
                print('Cost after {} iterations: {}'.format(i, cost))
        if graph:
            plt.plot(np.arange(0, iterations, step), costs)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """ Save the instance object to a file in pickle format

        Args:
            filename (_type_): _description_
        """
        if not filename.endswith(".pkl"):
            filename += ".pkl"
=======
        """
        getter function for weights
        Returns a dictionary to hold all weights and biased of the network
        """
        return self.__weights

    @property
    def activation(self):
        """
        getter function for activation
        Returns a string with the activation function for hidden layers
        """
        return self.__activation

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        Arguments:
         - X (numpy.ndarray): with shape (nx, m) that contains the input data
           * nx is the number of input features to the neuron
           * m is the number of examples
        """
        self.__cache['A0'] = X

        for i in range(self.__L):
            wkey = "W{}".format(i + 1)
            bkey = "b{}".format(i + 1)
            Aprevkey = "A{}".format(i)
            Akey = "A{}".format(i + 1)
            W = self.__weights[wkey]
            b = self.__weights[bkey]
            Aprev = self.__cache[Aprevkey]

            z = np.matmul(W, Aprev) + b
            if i < self.__L - 1:
                if self.__activation == 'sig':
                    self.__cache[Akey] = self.sigmoid(z)
                else:
                    self.__cache[Akey] = np.tanh(z)
            else:
                self.__cache[Akey] = self.softmax(z)

        return (self.__cache[Akey], self.__cache)

    def sigmoid(self, z):
        """
        Applies the sigmoid activation function
        Arguments:
        - z (numpy.ndattay): with shape (nx, m) that contains the input data
         * nx is the number of input features to the neuron.
         * m is the number of examples
        Updates the private attribute __A
        The neuron should use a sigmoid activation function
        Return:
        The private attribute A
        """
        y_hat = 1 / (1 + np.exp(-z))
        return y_hat

    def softmax(self, z):
        """
        Applies the softmax activation function
        Arguments:
        - z (numpy.ndattay): with shape (nx, m) that contains the input data
         * nx is the number of input features to the neuron.
         * m is the number of examples
        Updates the private attribute __A
        The neuron should use a sigmoid activation function

        Return:
        The private attribute y_hat
        """
        y_hat = np.exp(z - np.max(z))
        return y_hat / y_hat.sum(axis=0)

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Arguments:
         - Y (numpy.ndarray): with shape (1, m) that contains the correct
                              labels for the input data
         - A (numpy.ndarray): with shape (1, m) containing the activated output
                              of the neuron for each example
        Returns:
         The cost
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A)) / m

        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network’s predictions
        Arguments:
         - X is a numpy.ndarray with shape (nx, m) that contains the input data
           * nx is the number of input features to the neuron
           * m is the number of examples
         - Y (numpy.ndarray): with shape (1, m) that contains the correct
             labels for the input data
        Returns:
         The neuron’s prediction and the cost of the network, respectively
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        Y_hat = np.max(A, axis=0)
        A = np.where(A == Y_hat, 1, 0)
        return A, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        - Calculates one pass of gradient descent on the neural network.
        - Y is a numpy.ndarray with shape (1, m) that contains the
        correct labels for the input data.
        - cache is a dictionary containing all the intermediary
        values of the network.
        - alpha is the learning rate.
        - Updates the private attribute __weights.
        """
        # start the backpropagation
        m = Y.shape[1]
        # dA = - (np.divide(Y, A) - np.divide(1 - Y, 1 - A))
        weights_c = self.__weights.copy()
        for i in range(self.__L, 0, -1):
            A = cache["A" + str(i)]
            if i == self.__L:
                dz = A - Y
            else:
                if self.__activation == "sig":
                    g = A * (1 - A)
                    dz = (weights_c["W" + str(i + 1)].T @ dz) * g
                elif self.__activation == "tanh":
                    g = 1 - (A ** 2)
                    dz = (weights_c["W" + str(i + 1)].T @ dz) * g
            dw = (dz @ cache["A" + str(i - 1)].T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            # dz for next iteration
            self.__weights["W" + str(i)] = self.__weights[
                    "W" + str(i)] - (alpha * dw)
            self.__weights["b" + str(i)] = self.__weights[
                    "b" + str(i)] - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the deep neural network by updating the private attributes
        Arguments:
         - X (numpy.ndarray): with shape (nx, m) that contains the input data
           * nx is the number of input features to the neuron
           * m is the number of examples
         - Y (numpy.ndarray):  with shape (1, m) that contains the correct
              labels for the input data
         - iterations (int): is the number of iterations to train over
         - alpha (float): is the learning rate
         - varbose (boolean): that defines whether or not to print
              information about the training
         - graph (boolean): that defines whether or not to graph information
              about the training once the training has completed
        """

        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step >= iterations:
                raise ValueError("step must be positive and <= iterations")

        cost_list = []
        step_list = []
        for i in range(iterations):
            A, self.__cache = self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)
            cost = self.cost(Y, A)
            cost_list.append(cost)
            step_list.append(i)
            if verbose and i % step == 0:
                print("Cost after {} iterations: {}".format(i, cost))

        if graph:
            plt.plot(step_list, cost_list)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title("Trainig Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format

        Arguments:
        - filename is the file to which the object should be saved
        If filename does not have the extension .pkl, add it

        """
        if not filename.endswith(".pkl"):
            filename = filename + ".pkl"
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
<<<<<<< HEAD
        """ Load a pickled DeepNeuralNetwork object

        Args:
            filename (_type_): _description_

        Returns:
            _type_: _description_
=======
        """
        Loads a pickled DeepNeuralNetwork object

        Arguments:
        - filename is the file from which the object should be loaded

        Returns:
        The loaded object, or None if filename doesn’t exist
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
        """
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
