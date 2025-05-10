#!/usr/bin/env python3
<<<<<<< HEAD
'''
    calculate descriptive statistics
    for all columns in pd.DataFrame except Timestamp
'''

import pandas as pd


=======
"""
New code calculates descriptive statistics for all columns except Timestamp
"""

import pandas as pd
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

<<<<<<< HEAD
stats = df.describe()
=======
stats = df.drop(columns=['Timestamp']).describe()
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6

print(stats)
