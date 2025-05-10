#!/usr/bin/env python3
<<<<<<< HEAD
'''
    Based on 11-concat.py, rearrange the MultiIndex
    levels such that timestamp is the first level:

    TODO:
    - Concatenate the bitstamp and coinbase tables from timestamps
    1417411980 to 1417417980, inclusive
    - Add keys to the data labeled bitstamp and
    coinbase respectively
    - Display the rows in chronological order

'''
=======
"""
New code rearranges the MultiIndex levels such that the timestamp is first
"""
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6

import pandas as pd
from_file = __import__('2-from_file').from_file

<<<<<<< HEAD
# Load data
df1 = from_file('../Data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df2 = from_file('../Data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', ',')

# Set 'Timestamp' as index
df1 = df1.set_index('Timestamp')
df2 = df2.set_index('Timestamp')

# Filter the dataframes by the specified timestamps
df1 = df1.loc[1417411980:1417417980]
df2 = df2.loc[1417411980:1417417980]

# Add keys to the data
df1['key'] = 'coinbase'
df2['key'] = 'bitstamp'

# Reset index to add 'Timestamp' as a column
df1 = df1.reset_index()
df2 = df2.reset_index()

# Set new MultiIndex with 'Timestamp' first and 'key' second
df1 = df1.set_index(['Timestamp', 'key'])
df2 = df2.set_index(['Timestamp', 'key'])

# Concatenate the dataframes
df = pd.concat([df1, df2])

# Sort the DataFrame by the new MultiIndex
df = df.sort_index()

# Print the DataFrame
=======
df1 = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df2 = from_file('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', ',')

df1 = df1.loc[
    (df1['Timestamp'] >= 1417411980) & (df1['Timestamp'] <= 1417417980)]
df2 = df2.loc[
    (df2['Timestamp'] >= 1417411980) & (df2['Timestamp'] <= 1417417980)]

df1 = df1.set_index('Timestamp')
df2 = df2.set_index('Timestamp')

df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])

df = df.reorder_levels([1, 0], axis=0)

df = df.sort_index()

>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
print(df)
