#!/usr/bin/env python3
<<<<<<< HEAD
'''
    remove the entries in the pd.DataFrame
    where Close is NaN:
'''
=======
"""
New code removes the entries in the DataFrame where Close is NaN
"""
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6

import pandas as pd
from_file = __import__('2-from_file').from_file

<<<<<<< HEAD
df = from_file('../Data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
=======
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6

df = df.dropna(subset=['Close'])

print(df.head())
