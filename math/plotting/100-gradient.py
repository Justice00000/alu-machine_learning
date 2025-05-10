#!/usr/bin/env python3
<<<<<<< HEAD
""" plot mountain elevation as a scatter plot with colorbar """
=======
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)

x = np.random.randn(2000) * 10
y = np.random.randn(2000) * 10
z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))

<<<<<<< HEAD
plt.scatter(x, y, c=z)
plt.colorbar(label="elevation (m)")
plt.xlabel('x coordinate (m)')
plt.ylabel('y coordinate (m)')
plt.title("Mountain Elevation")
=======
# your code here
plt.scatter(x, y, c=z)
plt.colorbar(label="elevation (m)", orientation="vertical") 
plt.xlabel('x coordinate (m)')
plt.ylabel('y coordinate (m)')
plt.title('Mountain Elevation')
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
plt.show()
