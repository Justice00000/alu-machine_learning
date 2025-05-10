#!/usr/bin/env python3
<<<<<<< HEAD
""" plot the given PCA data as a 3D scatter plot """
=======
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

<<<<<<< HEAD
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2],
           c=labels, cmap=plt.get_cmap('plasma'), label=labels)
ax.set_xlabel('U1')
ax.set_ylabel('U2')
ax.set_zlabel('U3')
plt.title("PCA of Iris Dataset")
=======
# your code here
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.scatter([x[0] for x in pca_data], [y[1] for y in pca_data], [z[2] for z in pca_data], c=labels, cmap='plasma')
plt.title('PCA of Iris Dataset')
ax.set_xlabel('U1')
ax.set_ylabel('U2')
ax.set_zlabel('U3')
# plt.tight_layout()
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
plt.show()
