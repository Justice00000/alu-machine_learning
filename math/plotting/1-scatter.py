#!/usr/bin/env python3
<<<<<<< HEAD
""" plots x, y as a scatter plot """
=======
import os
>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c
import numpy as np
import matplotlib.pyplot as plt

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x, y = np.random.multivariate_normal(mean, cov, 2000).T
y += 180

<<<<<<< HEAD
plt.scatter(x, y, c='m')
plt.xlabel('Height (in)')
plt.ylabel('Weight (lbs)')
plt.title("Men's Height vs Weight")
plt.show()
=======
plt.scatter(x, y, c='magenta', s=10)
plt.title('Men\'s Height vs Weight')
plt.xlabel('Height (in)')
plt.ylabel('Weight (lbs)')
plt.savefig(
    f"plots/{os.path.basename(__file__)[0:-3] + '_plot.png'}"
)
plt.show()
>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c
