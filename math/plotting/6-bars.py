#!/usr/bin/env python3
<<<<<<< HEAD
""" plot a stacked bar chart for the amount and types of fruit people have """
=======
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
<<<<<<< HEAD
fruit = np.random.randint(0, 20, (4, 3))

people = ['Farrah', 'Fred', 'Felicia']
fruit_names = {
    'apples': 'red',
    'bananas': 'yellow',
    'oranges': '#ff8000',
    'peaches': '#ffe5b4'
}

i = 0
for name, color in sorted(fruit_names.items()):
    bottom = 0
    for j in range(i):
        bottom += fruit[j]
    plt.bar(
        np.arange(len(people)),
        fruit[i],
        width=0.5,
        bottom=bottom,
        color=color,
        label=name)
    i += 1
plt.xticks(np.arange(len(people)), people)
plt.yticks(np.arange(0, 81, 10))
plt.ylabel('Quantity of Fruit')
plt.title("Number of Fruit per Person")
plt.legend()
=======
fruit = np.random.randint(0, 20, (4,3))


# your code here
fig, ax = plt.subplots()
bottom = np.zeros(3)

names = ["Farrah", "Fred", "Felicia"]
fruits = ['apples', 'bananas', 'oranges', 'peaches']

width=0.5

colors = {
    0: 'red',
    1: 'yellow',
    2: '#ff8000',
    3: '#ffe5b4'
}

for i in range(len(fruit)):
  label = fruits[i]
  val = fruit[i]
  ax.bar(names, val, bottom=bottom, label=label, color=colors[i], width=width)
  bottom += val

ax.set_yticks(range(0, 81, 10))
ax.set_ylabel('Quantity of Fruit')
ax.set_title('Number of Fruit per Person')
ax.legend(loc="upper right")
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
plt.show()
