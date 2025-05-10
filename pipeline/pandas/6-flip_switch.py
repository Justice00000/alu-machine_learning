#!/usr/bin/env python3
<<<<<<< HEAD
'''
     script to alter the pd.DataFrame such that the rows and
     columns are transposed and the data is sorted in reverse
     chronological order:
'''

import pandas as pd


=======
"""
New code transposes the rows and columns and then sorts the data
    in reverse chronological order
"""

import pandas as pd
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

<<<<<<< HEAD
df = df[::-1].T
=======
df = df.sort_values(by='Timestamp', ascending=False).T
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6

print(df.tail(8))
