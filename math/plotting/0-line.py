#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

plt.plot(y, 'r-')  # 'r-' specifies a red solid line
plt.xlim(0, 10)  # Set x-axis range from 0 to 10
plt.show()
