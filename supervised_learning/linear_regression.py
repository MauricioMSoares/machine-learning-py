import numpy as np
import matplotlib.pyplot as plt


height = np.array([5.2, 5.7, 5.9, 6.0, 6.4])
weight = np.array([125, 140, 155, 160, 180])
m, b = np.polyfit(height, weight, 1)
plt.scatter(height, weight)
plt.plot(height, m * height + b, color="red")
plt.xlabel("Height (in)")
plt.ylabel("Weight (lb)")
plt.show()

new_height = 5.8
predicted_weight = m * new_height + b
print("Predicted weight: ", predicted_weight)
