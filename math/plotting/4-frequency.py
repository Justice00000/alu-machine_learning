mport numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# Plotting the histogram
plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')  # Bins every 10 units, bars outlined in black

# Adding labels and title
plt.xlabel("Grades")
plt.ylabel("Number of Students")
plt.title("Project A")

# Display the plot
plt.show()

