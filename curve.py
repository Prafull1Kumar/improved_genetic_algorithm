import numpy as np
import matplotlib.pyplot as plt

def parabolic_curve(x):
    a = -0.01  # Coefficient controlling the shape of the parabola
    b = 1.0    # Coefficient controlling the vertical shift
    c = 0.0    # Coefficient controlling the horizontal shift
    return a * (x - c)**2 + b

x_values = np.linspace(0, 200, 100)  # Generate 100 x-values from 0 to 200
y_values = (x_values)*(x_values)/40000

plt.plot(x_values, y_values)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Parabolic Curve')
plt.grid(True)
plt.show()
