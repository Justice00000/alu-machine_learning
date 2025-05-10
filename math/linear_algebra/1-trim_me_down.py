#!/usr/bin/env python3
matrix = [[1, 3, 9, 4, 5, 8], [2, 4, 7, 3, 4, 0], [0, 3, 4, 6, 1, 5]]
the_middle = []
<<<<<<< HEAD
for i in range(len(matrix)):
    the_middle.append(matrix[i][2:4])
=======
for row in matrix:
    the_middle.append(row[2:4])
>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c
print("The middle columns of the matrix are: {}".format(the_middle))
