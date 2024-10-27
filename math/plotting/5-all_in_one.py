<<<<<<< HEAD
#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

=======
mport numpy as np
import matplotlib.pyplot as plt

# Data for the plots
>>>>>>> a262473352dd582d190fed95fa6e5c665f37f9ba
y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

<<<<<<< HEAD
fig = plt.figure()

plot1 = fig.add_subplot(3, 2, 1)
plot1.plot(y0, 'r-')
plot1.set_xlim((0, 10))


plot2 = fig.add_subplot(3, 2, 2)
plot2.scatter(x1, y1, c='magenta', s=10)
plot2.set_title('Men\'s Height vs Weight', fontsize='x-small')
plot2.set_xlabel('Height (in)', fontsize='x-small')
plot2.set_ylabel('Weight (lbs)', fontsize='x-small')


plot3 = fig.add_subplot(3, 2, 3)
plot3.plot(x2, y2)
plot3.set_title('Exponential Decay of C-14', fontsize='x-small')
plot3.set_xlabel('Time (years)', fontsize='x-small')
plot3.set_ylabel('Fraction Remaining', fontsize='x-small')
plot3.set_yscale('log')
plot3.set_xlim(0, 28650)


plot4 = fig.add_subplot(3, 2, 4)
plot4.plot(x3, y31, 'r--', label='C-14')
plot4.plot(x3, y32, 'g-', label='Ra-226')
plot4.set_xlabel("Time (years)", fontsize='x-small')
plot4.set_ylabel("Fraction Remaining", fontsize='x-small')
plot4.set_title(
    "Exponential Decay of Radioactive Elements",
    fontsize='x-small'
)
plot4.axis([0, 20000, 0, 1])
plot4.legend()


plot5 = fig.add_subplot(3, 1, 3)
plot5.set_xlabel('Grades', fontsize='x-small')
plot5.set_ylabel('Number of Students', fontsize='x-small')
plot5.set_title('Project A', fontsize='x-small')
plt.axis([0, 100, 0, 30])
plot5.hist(
    student_grades,
    bins=10,
    edgecolor='black',
    range=(0, 100)
)


fig.suptitle("All in One")
fig.tight_layout()
plt.savefig(
    f"plots/{os.path.basename(__file__)[0:-3] + '_plot.png'}"
)
plt.show()
=======
# Create a figure with a 3x2 grid
fig, axs = plt.subplots(3, 2, figsize=(12, 12))
fig.suptitle('All in One', fontsize='x-small')

# Plot 1: y0 as a line graph
axs[0, 0].plot(y0, 'r-', label='y0')
axs[0, 0].set_xlabel("X-axis")
axs[0, 0].set_ylabel("Y-axis")
axs[0, 0].set_title("Line Graph", fontsize='x-small')

# Plot 2: x1 ↦ y1 as a scatter plot
axs[0, 1].scatter(x1, y1, color='magenta', label="Men's Height vs Weight")
axs[0, 1].set_xlabel("Height (in)")
axs[0, 1].set_ylabel("Weight (lbs)")
axs[0, 1].set_title("Scatter Plot", fontsize='x-small')

# Plot 3: x2 ↦ y2 as a line graph with logarithmic y-axis
axs[1, 0].plot(x2, y2, 'b-', label='Exponential Decay')
axs[1, 0].set_xlabel("Time (years)")
axs[1, 0].set_ylabel("Fraction Remaining")
axs[1, 0].set_title("Exponential Decay", fontsize='x-small')
axs[1, 0].set_yscale('log')
axs[1, 0].set_xlim(0, 20000)

# Plot 4: x3 ↦ y31 and x3 ↦ y32 as line graphs
axs[1, 1].plot(x3, y31, 'r--', label="C-14")
axs[1, 1].plot(x3, y32, 'g-', label="Ra-226")
axs[1, 1].set_xlabel("Time (years)")
axs[1, 1].set_ylabel("Fraction Remaining")
axs[1, 1].set_title("Decay Comparison", fontsize='x-small')
axs[1, 1].set_xlim(0, 20000)
axs[1, 1].set_ylim(0, 1)
axs[1, 1].legend(loc="upper right")

# Plot 5: Histogram of student grades
axs[2, 0].hist(student_grades, bins=range(0, 101, 10), edgecolor='black')
axs[2, 0].set_xlabel("Grades")
axs[2, 0].set_ylabel("Number of Students")
axs[2, 0].set_title("Project A", fontsize='x-small')

# Remove empty subplot (2, 1)
fig.delaxes(axs[2, 1])

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Show the plot
plt.show()

>>>>>>> a262473352dd582d190fed95fa6e5c665f37f9ba
