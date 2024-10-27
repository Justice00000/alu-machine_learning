<<<<<<< HEAD
#!/usr/bin/env python3
import os
import numpy as np
=======
mport numpy as np
>>>>>>> a262473352dd582d190fed95fa6e5c665f37f9ba
import matplotlib.pyplot as plt

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x, y = np.random.multivariate_normal(mean, cov, 2000).T
y += 180

<<<<<<< HEAD
plt.scatter(x, y, c='magenta', s=10)
plt.title('Men\'s Height vs Weight')
plt.xlabel('Height (in)')
plt.ylabel('Weight (lbs)')
plt.savefig(
    f"plots/{os.path.basename(__file__)[0:-3] + '_plot.png'}"
)
plt.show()
=======
# Plotting the data as a scatter plot
plt.scatter(x, y, color='magenta')  # Plot data as magenta points

# Adding labels and title
plt.xlabel("Height (in)")
plt.ylabel("Weight (lbs)")
plt.title("Men's Height vs Weight")

# Display the plot
plt.show()

>>>>>>> a262473352dd582d190fed95fa6e5c665f37f9ba
