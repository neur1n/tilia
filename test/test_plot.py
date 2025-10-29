#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# Example Data: N runs of 3 rows (labels) by 4 columns (features)
N = 10  # Number of runs
data = [np.random.rand(3, 4) for _ in range(N)]  # Random stability values

# Step 1: Combine stability values for each feature and label
all_data = np.array(data)  # Shape: (N, 3, 4)
grouped_data = {}  # Dictionary to store data for each feature and label

for feature_idx in range(4):  # 4 features
    grouped_data[f"Feature {feature_idx+1}"] = [
        all_data[:, label_idx, feature_idx].flatten() for label_idx in range(3)
    ]
breakpoint()

# Step 2: Prepare data for plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Boxplot settings
positions = []  # X-axis positions for boxplots
box_data = []  # Stability values for all boxplots
group_width = 0.8  # Total width of each group
num_labels = 3  # 3 boxplots per feature
offset = group_width / num_labels

for feature_idx, (feature_name, feature_data) in enumerate(grouped_data.items()):
    # Add data for each label
    for label_idx, label_data in enumerate(feature_data):
        box_data.append(label_data)
        positions.append(feature_idx + label_idx * offset)

# Step 3: Plot the boxplots
plt.boxplot(box_data, positions=positions, widths=offset * 0.8)

# Step 4: Format the plot
# Adjust x-axis ticks to show grouped features
plt.xticks(
    [i + group_width / 2 - offset / 2 for i in range(4)],
    [f"Feature {i+1}" for i in range(4)],
)
plt.title("Stability by Feature and Label")
plt.xlabel("Features")
plt.ylabel("Stability")
plt.grid(axis="y")
plt.legend(
    [f"Label {i+1}" for i in range(3)],
    loc="upper right",
    title="Labels",
    bbox_to_anchor=(1.1, 1),
)
plt.show()
