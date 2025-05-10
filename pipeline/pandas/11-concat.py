#!/usr/bin/env python3
<<<<<<< HEAD
'''
    index the pd.DataFrames on the Timestamp
    columns and concatenate them:
'''
=======
"""
New code indexes the DataFrame on the Timestamp columns and concatenates them
"""
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6

import pandas as pd
from_file = __import__('2-from_file').from_file

df1 = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df2 = from_file('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', ',')

<<<<<<< HEAD
=======
df2 = df2.loc[df2['Timestamp'] <= 1417411920]
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6

df1 = df1.set_index('Timestamp')
df2 = df2.set_index('Timestamp')

<<<<<<< HEAD
df = pd.concat([df1, df2])
=======
df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6

print(df)
