#!/usr/bin/env python3
<<<<<<< HEAD
'''
    sort the pd.DataFrame by the High
    price in descending order:
'''
=======
"""
New code sorts the DataFrame by the High price in descending order
"""
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6

import pandas as pd
from_file = __import__('2-from_file').from_file

<<<<<<< HEAD
df = from_file('../Data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.sort_values(by='High', ascending=False)
=======
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.sort_values(by='High', ascending=False)

>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
print(df.head())
