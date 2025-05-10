#!/usr/bin/env python3
<<<<<<< HEAD
""" plots y as a line graph """
=======
import os
>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

plt.plot(y, 'r-')
<<<<<<< HEAD
plt.xlim((0, 10))
plt.show()
=======
plt.savefig(
    f"plots/{os.path.basename(__file__)[0:-3] + '_plot.png'}"
)
plt.show()
>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c
