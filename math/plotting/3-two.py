#!/usr/bin/env python3
<<<<<<< HEAD
""" plots x, y1 and x, y2 as line graphs, each with their own formatting """
=======
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)

<<<<<<< HEAD
plt.plot(x, y1, 'r--', label='C-14')
plt.plot(x, y2, 'g-', label='Ra-226')
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.title("Exponential Decay of Radioactive Elements")
plt.legend()
plt.xlim((0, 20000))
plt.ylim((0, 1))
=======
# your code here
plt.plot(x, y1, "r--", label="C-14")
plt.plot(x, y2, "g", label="Ra-226")
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.title('Exponential Decay of Radioactive Elements')
plt.legend(loc="upper right")
plt.axis([0 , 20000, 0, 1])
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
plt.show()
