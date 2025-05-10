#!/usr/bin/env python3
<<<<<<< HEAD
'''
    function def from_file(filename, delimiter):
    that loads data from a file as a pd.DataFrame:
'''
=======
"""
Defines function that loads data from a file as a Pandas DataFrame
"""
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6


import pandas as pd


def from_file(filename, delimiter):
<<<<<<< HEAD
    '''
        Args:
            - filename is the file to load from
            - delimiter is the column separator

        Returns:
            - Returns: the loaded pd.DataFrame
    '''

    return pd.read_csv(filename, delimiter=delimiter)
=======
    """
    Loads data from a file as a Pandas DataFrame

    parameters:
        filename [str]: file to load the data from
        delimiter [str]: the column separator

    returns:
        the newly created pd.DataFrame
    """
    df = pd.read_csv(filename, delimiter=delimiter)
    return df
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
