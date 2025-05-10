#!/usr/bin/env python3
<<<<<<< HEAD
""" plot a stacked bar chart for the amount and types of fruit people have """
=======
import os
>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))
<<<<<<< HEAD

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
plt.show()
=======
owners = ['Farrah', 'Fred', 'Felicia']
fruit_names_and_colors = (
    ('apples', 'red'),
    ('bananas', 'yellow'),
    ('oranges', '#ff8000'),
    ('peaches', '#ffe5b4')
)

buttom_spacing = np.zeros(fruit.shape[1])
for i in range(len(fruit)):
    plt.bar(
        owners,
        fruit[i],
        bottom=buttom_spacing,
        width=0.5,
        label=fruit_names_and_colors[i][0],
        color=fruit_names_and_colors[i][1],
    )
    buttom_spacing += fruit[i, :]

plt.ylim(0, 80)
plt.title('Number of Fruit per Person')
plt.xlabel('Fruit Owners')
plt.ylabel('Quantity of Fruit')
plt.legend()
plt.savefig(
    f"plots/{os.path.basename(__file__)[0:-3] + '_plot.png'}"
)
plt.show()
>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c
