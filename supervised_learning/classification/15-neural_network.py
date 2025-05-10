#!/usr/bin/env python3
<<<<<<< HEAD
""" Neural Network
"""

import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
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
File defines a class that represents a
neural network with one hidden layer
"""


import numpy as np


class NeuralNetwork:
    """
    Class that represents a neural network with one hidden layer
    """

    def __init__(self, nx, nodes):
        """
        Initializes a neural network with one hidden layer
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

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression

        Args:
            Y (_type_): _description_
            A (_type_): _description_
        """
        loss = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost = np.mean(loss)
        return cost

    def evaluate(self, X, Y):
        """ Evaluates the neural network’s predictions

        Args:
            X (_type_): _description_
            Y (_type_): _description_
        """
        self.forward_prop(X)
        return np.where(self.__A2 >= 0.5, 1, 0), self.cost(Y, self.__A2)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """ Calculates one pass of gradient descent on the neural network

        Args:
            X (_type_): _description_
            Y (_type_): _description_
            A1 (_type_): _description_
            A2 (_type_): _description_
            alpha (float, optional): _description_. Defaults to 0.05.
        """
        m = Y.shape[1]
        dz2 = A2 - Y
        dw2 = np.matmul(A1, dz2.T) / m
        db2 = np.sum(dz2, axis=1, keepdims=True) / m
        dz1 = np.matmul(self.__W2.T, dz2) * A1 * (1 - A1)
        dw1 = np.matmul(X, dz1.T) / m
        db1 = np.sum(dz1, axis=1, keepdims=True) / m
        self.__W2 -= alpha * dw2.T
        self.__b2 -= alpha * db2
        self.__W1 -= alpha * dw1.T
        self.__b1 -= alpha * db1

    def train(self, X, Y, iterations=1000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """ Trains the neural network

        Args:
            X (_type_): _description_
            Y (_type_): _description_
            iterations (int, optional): _description_. Defaults to 5000.
            alpha (float, optional): _description_. Defaults to 0.05.
            verbose (bool, optional): _description_. Defaults to True.
            graph (bool, optional): _description_. Defaults to True.
            step (int, optional): _description_. Defaults to 100.
        """
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations < 1:
            raise ValueError('iterations must be a positive integer')

        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')

        if graph or verbose:
            if not isinstance(step, int):
                raise TypeError('step must be an integer')
            if step < 1 or step > iterations:
                raise ValueError('step must be positive and <= iterations')

        costs = []
        steps = []
        for i in range(iterations + 1):
            self.forward_prop(X)
            cost = self.cost(Y, self.__A2)
            if i % step == 0:
                if verbose:
                    print('Cost after {} iterations: {}'.format(i, cost))
                if graph:
                    costs.append(cost)
                    steps.append(i)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
        if graph:
            plt.plot(steps, costs, 'b')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)
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
        Computes the forward propagation of the neural network
        """
        z1 = np.matmul(self.W1, X) + self.b1
        self.__A1 = 1 / (1 + (np.exp(-z1)))
        z2 = np.matmul(self.W2, self.__A1) + self.b2
        self.__A2 = 1 / (1 + (np.exp(-z2)))
        return (self.A1, self.A2)

    def cost(self, Y, A):
        """
        Computes the cost of the neural network
        """
        m = Y.shape[1]
        m_loss = np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A)))
        cost = (1 / m) * (-(m_loss))
        return (cost)

    def evaluate(self, X, Y):
        """
        Calculates the prediction and cost of the neural network
        """
        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        prediction = np.where(A2 >= 0.5, 1, 0)
        return (prediction, cost)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Computes one pass of the gradient descent on the neural network
        """
        m = Y.shape[1]
        dz2 = (A2 - Y)
        d__W2 = (1 / m) * (np.matmul(dz2, A1.transpose()))
        d__b2 = (1 / m) * (np.sum(dz2, axis=1, keepdims=True))
        dz1 = (np.matmul(self.W2.transpose(), dz2)) * (A1 * (1 - A1))
        d__W1 = (1 / m) * (np.matmul(dz1, X.transpose()))
        d__b1 = (1 / m) * (np.sum(dz1, axis=1, keepdims=True))
        self.__W1 = self.W1 - (alpha * d__W1)
        self.__b1 = self.b1 - (alpha * d__b1)
        self.__W2 = self.W2 - (alpha * d__W2)
        self.__b2 = self.b2 - (alpha * d__b2)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Training Function for the neural network
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        if graph:
            import matplotlib.pyplot as plt
            x_points = np.arange(0, iterations + 1, step)
            points = []
        for itr in range(iterations):
            A1, A2 = self.forward_prop(X)
            if verbose and (itr % step) == 0:
                cost = self.cost(Y, A2)
                print("Cost after " + str(itr) + " iterations: " + str(cost))
            if graph and (itr % step) == 0:
                cost = self.cost(Y, A2)
                points.append(cost)
            self.gradient_descent(X, Y, A1, A2, alpha)
        itr += 1
        if verbose:
            cost = self.cost(Y, A2)
            print("Cost after " + str(itr) + " iterations: " + str(cost))
        if graph:
            cost = self.cost(Y, A2)
            points.append(cost)
            y_points = np.asarray(points)
            plt.plot(x_points, y_points, 'b')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()
        return (self.evaluate(X, Y))
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
