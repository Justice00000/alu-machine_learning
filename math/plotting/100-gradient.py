#!/usr/bin/env python3
<<<<<<< HEAD
""" plot mountain elevation as a scatter plot with colorbar """
=======
import os
>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c
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
plt.show()
=======

plt.scatter(x, y, c=z)
plt.xlabel('x coordinate (m)')
plt.ylabel('y coordinate (m)')
plt.title('Mountain Elevation')
cbar = plt.colorbar()
cbar.set_label('elevation (m)')
plt.savefig(
    f"plots/{os.path.basename(__file__)[0:-3] + '_plot.png'}"
)
plt.show()
>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c
