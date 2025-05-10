#!/usr/bin/env python3
<<<<<<< HEAD
=======
import os
>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

<<<<<<< HEAD
plt.hist(student_grades,
         bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
         edgecolor='black')
plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.title("Project A")
plt.xlim((0, 100))
plt.xticks(np.arange(0, 101, 10))
plt.ylim((0, 30))
plt.show()
=======
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
>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c
