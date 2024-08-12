import matplotlib.pyplot as plt
import numpy as np

# Sample data
labels = ['A', 'B', 'C']
value1 = [50, 80, 70]
value2 = [60, 70, 90]
labels_inside = ['Label 1', 'Label 2', 'Label 3']

# Plotting the double bar graph
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, value1, width, label='Value 1')
rects2 = ax.bar(x + width/2, value2, width, label='Value 2')
rects3 = ax.bar(x + width/2, labels, width, label='Value 3')

# Adding labels above the bars
for i, rect in enumerate(rects1):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, str(value1[i]), ha='center', va='bottom')

for i, rect in enumerate(rects2):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, str(value2[i]), ha='center', va='bottom')

# Adding labels inside the bars
for i, rect in enumerate(rects1):
    width = rect.get_width()
    ax.text(rect.get_x() + width/2, rect.get_height()/2, labels_inside[i], ha='center', va='center', color='white')

for i, rect in enumerate(rects2):
    width = rect.get_width()
    ax.text(rect.get_x() + width/2, rect.get_height()/2, labels_inside[i], ha='center', va='center', color='white')

# Other plot configurations
ax.set_ylabel('Values')
ax.set_title('Double Bar Graph')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.show()
