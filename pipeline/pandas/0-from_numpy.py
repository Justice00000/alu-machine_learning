#!/usr/bin/env python3
<<<<<<< HEAD
'''
    Function def from_numpy(array):
    that creates a pd.DataFrame from a np.ndarray
'''


import string
=======
"""
Defines function that creates a Pandas DataFrame from a Numpy ndarray
"""


>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
import pandas as pd


def from_numpy(array):
<<<<<<< HEAD
    '''
        Function def from_numpy(array):
        that creates a pd.DataFrame from a np.ndarray

        Args:
            - array is the np.ndarray from which you should
            create the pd.DataFrame
            - The columns of the pd.DataFrame should be labeled
            in alphabetical order and capitalized.

        Returns:
            - Returns: the newly created pd.DataFrame
    '''
    num_columns = array.shape[1]

    # Generate the column labels (A, B, C, ...)
    columns = [chr(65 + i) for i in range(num_columns)]

    # Create the DataFrame
    df = pd.DataFrame(array, columns=columns)

=======
    """
    Creates a Pandas DataFrame from a numpy.ndarray

    parameters:
        array [numpy.ndarray]: array to create pd.DataFrame from

    columns of the DataFrame should be labeled in alphabetical order
        and capitalized (there will not be more than 26 columns)

    returns:
        the newly created pd.DataFrame
    """
    alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I",
                "J", "K", "L", "M", "N", "O", "P", "Q", "R",
                "S", "T", "U", "V", "W", "X", "Y", "Z"]
    column_labels = []
    for i in range(len(array[0])):
        column_labels.append(alphabet[i])
    df = pd.DataFrame(array, columns=column_labels)
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
    return df
