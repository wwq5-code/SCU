import matplotlib.pyplot as plt

# Example data
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

plt.plot(x, y, marker='o')  # Plot the data

# Suppose you want to annotate the point at x=3, y=5
target_x = 3
target_y = 5
plt.text(target_x, target_y, f'({target_x}, {target_y})', fontsize=9, ha='right')

plt.show()
