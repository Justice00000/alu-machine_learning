mport numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

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

