#!/usr/bin/env python3
matrix = [[1, 3, 9, 4, 5, 8], [2, 4, 7, 3, 4, 0], [0, 3, 4, 6, 1, 5]]
the_middle = []
<<<<<<< HEAD
for i in range(len(matrix)):
    the_middle.append(matrix[i][2:4])
=======
# your code here
the_middle = [arr[2:4] for arr in matrix]
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
print("The middle columns of the matrix are: {}".format(the_middle))
