#!/usr/bin/env python3
arr = [9, 8, 2, 3, 9, 4, 1, 0, 3]
<<<<<<< HEAD
arr1 = arr[0:2] #First Two Elements
arr2 = arr[-5:] #Last Five Elements
arr3 = arr[1:6] #Index 1 Elements
=======
arr1 = arr[:2]
arr2 = arr[-5:]
arr3 = arr[1:6]
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
print("The first two numbers of the array are: {}".format(arr1))
print("The last five numbers of the array are: {}".format(arr2))
print("The 2nd through 6th numbers of the array are: {}".format(arr3))
