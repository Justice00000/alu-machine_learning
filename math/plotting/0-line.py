#!/usr/bin/env python3
<<<<<<< HEAD
""" plots y as a line graph """
=======
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

<<<<<<< HEAD
plt.plot(y, 'r-')
plt.xlim((0, 10))
=======
# your code here
plt.plot( y, 'r')
plt.xticks(range(0, 11, 2))
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
plt.show()
