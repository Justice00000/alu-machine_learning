#!/usr/bin/env python3
<<<<<<< HEAD
'''
    script to index the pd.DataFrame on
    the Timestamp column:
'''
=======
"""
New code indexes the DataFrame on the Timestamp column
"""
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6

import pandas as pd
from_file = __import__('2-from_file').from_file

<<<<<<< HEAD
df = from_file('../Data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
=======
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6

df = df.set_index('Timestamp')

print(df.tail())
