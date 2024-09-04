mport numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)

# Plotting the graphs
plt.plot(x, y1, 'r--', label="C-14")  # Dashed red line for y1 (C-14)
plt.plot(x, y2, 'g-', label="Ra-226")  # Solid green line for y2 (Ra-226)

# Adding labels, title, and legend
plt.xlabel("Time (years)")
plt.ylabel("Fraction Remaining")
plt.title("Exponential Decay of Radioactive Elements")

# Setting the axis ranges
plt.xlim(0, 20000)
plt.ylim(0, 1)

# Adding the legend in the upper right corner
plt.legend(loc="upper right")

# Display the plot
plt.show()

