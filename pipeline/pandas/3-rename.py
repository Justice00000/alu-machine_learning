#!/usr/bin/env python3
<<<<<<< HEAD
=======
"""
New code updates Pandas DataFrame script to:
- rename the column Timestamp to Datetime
- convert the timestamp values into datetime values
- display only the Datetime and Close columns
"""
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6

import pandas as pd
from_file = __import__('2-from_file').from_file

<<<<<<< HEAD
df = from_file('../Data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

#  Rename  Timestamp to Datetime
df = df.rename(columns={'Timestamp': 'Datetime'})
=======
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.rename(columns={'Timestamp': 'Datetime'})
df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
df = df.loc[:, ['Datetime', 'Close']]
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6

print(df.tail())
