<<<<<<< HEAD
#!/usr/bin/env python3
import os
import numpy as np
=======
mport numpy as np
>>>>>>> a262473352dd582d190fed95fa6e5c665f37f9ba
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))
<<<<<<< HEAD
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
=======

# Labels and colors for the plot
labels = ['Farrah', 'Fred', 'Felicia']
fruit_names = ['Apples', 'Bananas', 'Oranges', 'Peaches']
colors = {'Apples': 'red', 'Bananas': 'yellow', 'Oranges': '#ff8000', 'Peaches': '#ffe5b4'}

# Plotting the stacked bar graph
fig, ax = plt.subplots()

bottoms = np.zeros(fruit.shape[1])  # To stack bars on top of each other

for i, (fruit_type, color) in enumerate(colors.items()):
    ax.bar(labels, fruit[i, :], width=0.5, color=color, label=fruit_type, bottom=bottoms)
    bottoms += fruit[i, :]

# Adding labels, title, and legend
ax.set_ylabel("Quantity of Fruit")
ax.set_ylim(0, 80)
ax.set_yticks(np.arange(0, 81, 10))
ax.set_title("Number of Fruit per Person")
ax.legend(loc='upper right')

# Display the plot
plt.show()

>>>>>>> a262473352dd582d190fed95fa6e5c665f37f9ba
