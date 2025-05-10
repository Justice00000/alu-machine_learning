#!/usr/bin/env python3
<<<<<<< HEAD
""" plot the given PCA data as a 3D scatter plot """
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]
=======
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = np.load("./datasets/pca.npz/data.npy")
labels = np.load("./datasets/pca.npz/labels.npy")

>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

<<<<<<< HEAD
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2],
           c=labels, cmap=plt.get_cmap('plasma'), label=labels)
=======

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter(
    pca_data[:, 0],
    pca_data[:, 1],
    pca_data[:, 2],
    c=labels,
    label=labels,
    cmap=plt.get_cmap('plasma'),
)
>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c
ax.set_xlabel('U1')
ax.set_ylabel('U2')
ax.set_zlabel('U3')
plt.title("PCA of Iris Dataset")
<<<<<<< HEAD
=======

plt.tight_layout()
plt.savefig(
    f"plots/{os.path.basename(__file__)[0:-3] + '_plot.png'}"
)
>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c
plt.show()
