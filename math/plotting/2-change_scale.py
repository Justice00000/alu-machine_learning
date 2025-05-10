#!/usr/bin/env python3
<<<<<<< HEAD
""" plots x, y as a line graph where y-axis is scaled logarithmically """
=======
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)

<<<<<<< HEAD
plt.plot(x, y)
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.title("Exponential Decay of C-14")
plt.yscale("log")
plt.xlim((0, 28650))
=======
# your code here
# plt.plot()
plt.plot(x, y)
plt.yscale('log')
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.title('Exponential Decay of C-14')
plt.xlim(0, 28650)
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
plt.show()
