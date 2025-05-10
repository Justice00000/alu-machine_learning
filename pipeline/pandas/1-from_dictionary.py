#!/usr/bin/env python3
"""
<<<<<<< HEAD
    python script that created a
    pd.DataFrame from a dictionary:
=======
Creates a Pandas DataFrame from a dictionary and saves it into variable df
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
"""


import pandas as pd


<<<<<<< HEAD
"""
    Function def from_dictionary():
    that creates a pd.DataFrame from a dictionary

    Args:
    - The first column should be labeled First and have the
    values 0.0, 0.5, 1.0, and 1.5
    - The second column should be labeled Second and have the
    values one, two, three, four
    - The rows should be labeled A, B, C, and D, respectively

    Returns:
    - The pd.DataFrame should be saved into the variable df
"""

# Create the dictionary
dictionary = {
    'First': [0.0, 0.5, 1.0, 1.5],
    'Second': ['one', 'two', 'three', 'four']
}

# Create the DataFrame
df = pd.DataFrame(dictionary, index=['A', 'B', 'C', 'D'])
=======
def from_dictionary():
    """
    Creates a Pandas DataFrame from a dictionary

    The first column should be labeled First and have the values
        0.0, 0.5, 1.0, and 1.5.
    The second column should be labeled Second and hace the values
        one, two, three, four.
    The rows should be labeled A, B, C, and D, respectively.

    returns:
        the newly created pd.DataFrame
    """
    df = pd.DataFrame(
        {
            "First": [0.0, 0.5, 1.0, 1.5],
            "Second": ["one", "two", "three", "four"]
        },
        index=list("ABCD"))
    return df


df = from_dictionary()
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
