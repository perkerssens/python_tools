''''
    Polynomial regression on data points in a Cartesian coordinate system
'''

import numpy as np
from matplotlib import pyplot as plt
import numpy.polynomial.polynomial as poly
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10,10))
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
ax.set_box_aspect((3,3,3))

degree = 2
points = 55

x = np.linspace(0,2,points)
y = np.sin(x) + np.random.random(points) * 0.1
z = np.linspace(0,2,points) + np.random.random(points) * 0.8
P = np.array([x, y, z]).T

discrete_vector = np.arange(0, int(P.shape[0]))

coefs_x = poly.polyfit(discrete_vector, x, degree)
coefs_y = poly.polyfit(discrete_vector, y, degree)
coefs_z = poly.polyfit(discrete_vector, z, degree)

x_filtered = poly.polyval(discrete_vector, coefs_x)
y_filtered = poly.polyval(discrete_vector, coefs_y)
z_filtered = poly.polyval(discrete_vector, coefs_z)

ax.scatter(x, y, z, s=50, marker='x', label='raw')
ax.scatter(np.median(x), np.median(y), np.median(z), c='red', label='raw center point')

ax.plot(x_filtered, y_filtered, z_filtered, label='polyfit')
ax.scatter(np.median(x_filtered), np.median(y_filtered), np.median(z_filtered), c='green', label='filtered center point')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
fig.legend()

plt.show()
