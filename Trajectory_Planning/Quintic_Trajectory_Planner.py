'''
    Quintic trajectory planner
'''

import numpy as np
from matplotlib import pyplot as plt

# time step
h = 4e-3

def calculate_quintic_trajectory(t0, q0, v0, a0, tf, qf, vf, af):
    # calculate coefficients
    M = np.array([[1, t0, t0**2, t0**3, t0**4, t0**5],
                  [0, 1, 2*t0, 3*t0**2, 4*t0**3, 5*t0**4],
                  [0, 0, 2, 6*t0, 12*t0**2, 20*t0**3],
                  [1, tf, tf**2, tf**3, tf**4, tf**5],
                  [0, 1, 2*tf, 3*tf**2, 4*tf**3, 5*tf**4],
                  [0, 0, 2, 6*tf, 12*tf**2, 20*tf**3]])

    conditions = np.array([[q0, v0, a0, qf, vf, af]]).T
    coeff = (np.linalg.inv(M) @ conditions).T.flatten()


    # setup empty arrays
    t = np.linspace(t0, tf, int(tf/h))
    q = np.empty((t.shape))
    qd = np.empty((t.shape))
    qdd = np.empty((t.shape))

    # calculate trajectory
    for i in range(0, t.shape[0]):
        q[i] = coeff[0] + coeff[1]*t[i] + coeff[2]*t[i]**2 + coeff[3]*t[i]**3 + coeff[4]*t[i]**4 + coeff[5]*t[i]**5
        qd[i] = coeff[1] + 2*coeff[2]*t[i] + 3*coeff[3]*t[i]**2 + 4*coeff[4]*t[i]**3 + 5*coeff[5]*t[i]**4
        qdd[i] = 2*coeff[2] + 6*coeff[3]*t[i] + 12*coeff[4]*t[i]**2 + 20*coeff[5]*t[i]**3

    return q, qd, qdd


# initial conditions
t0 = 0
# q0 = 0
v0 = 0
a0 = 0
x0 = 500
y0 = 100
z0 = 50



# final conditions
tf = 10
# qf = 2
vf = 20
af = 0
xf = 520
yf = 400
zf = -200


[x, xd, xdd] = calculate_quintic_trajectory(t0, x0, v0, a0, tf, xf, vf, af)
[y, yd, ydd] = calculate_quintic_trajectory(t0, y0, v0, a0, tf, yf, vf, af)
[z, zd, zdd] = calculate_quintic_trajectory(t0, z0, v0, a0, tf, zf, vf, af)


t = np.linspace(0, tf, x.shape[0])
#
# while x0+0.0001 < xf:
#     [x1, xd1, xdd1] = calculate_quintic_trajectory(t0, x0, v0, a0, tf, xf, vf, af)
#     x0 = x1[1]
#     print(x0)


# plot results
fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

ax1.plot(t, x)
ax1.set_title('Position [mm]')

ax2.plot(t, xd)
ax2.set_title('Velocity [mm/s]')

ax3.plot(t, xdd)
ax3.set_title('Acceleration [mm/s^2]')

plt.show()

