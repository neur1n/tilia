#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

# Example matrix with values -1, 0, and -0.9
matrix = np.array([
    [-1, -0.9, 0],
    [0, -1, -0.9],
    [-0.9, -1, 0]
])

# Visualize the matrix with a fixed color scale
plt.matshow(matrix, cmap='coolwarm', vmin=-1.0, vmax=1.0)
plt.colorbar()  # Add a colorbar to show the scale
plt.show()
