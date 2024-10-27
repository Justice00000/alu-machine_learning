<<<<<<< HEAD
#!/usr/bin/env python3
import os
import numpy as np
=======
mport numpy as np
>>>>>>> a262473352dd582d190fed95fa6e5c665f37f9ba
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

<<<<<<< HEAD
plt.xlabel("Grades")
plt.ylabel("Number of Students")
plt.title("Project A")
plt.hist(
    student_grades,
    bins=10,
    edgecolor='black',
    range=(0, 100)
)
plt.axis([0, 100, 0, 30])
plt.savefig(
    f"plots/{os.path.basename(__file__)[0:-3] + '_plot.png'}"
)
plt.show()
=======
# Plotting the histogram
plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')  # Bins every 10 units, bars outlined in black

# Adding labels and title
plt.xlabel("Grades")
plt.ylabel("Number of Students")
plt.title("Project A")

# Display the plot
plt.show()

>>>>>>> a262473352dd582d190fed95fa6e5c665f37f9ba
