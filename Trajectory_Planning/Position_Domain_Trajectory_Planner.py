''''
    Trajectory planning based on the thesis of M. de Graaf
    SENSOR-GUIDED ROBOTIC LASER WELDING
'''

import numpy as np
from matplotlib import pyplot as plt
import scipy.signal

_T_s_ = 4e-3
_v_max_ = 50.
_a_max_ = 5.

# def scale_vector(P_unsc, P_current, P_next):
#     s_n = abs(P_next-P_current) / abs(P_unsc)
#     return s_n * P_unsc

class trajectory_generator():
    def __init__(self):
        self.prev_s_n = 0
        self.prev_d_lambda = 0

    def _scale_vector(self, P_unsc, P_current, P_next):
        s_n = abs(P_next - P_current) / abs(P_unsc)
        return s_n * P_unsc

    def generate(self, P__1, P_0, P_1, P_2):
        # dP_0_unscaled = (P_1 + P__1) / 2
        # s_n =  np.linalg.norm(P_0 - P_1) / np.linalg.norm(dP_0_unscaled)
        # dP_0 = s_n * dP_0_unscaled
        #
        # dP_1_unscaled = (P_2 + P_0) / 2
        # s_n_1 = np.linalg.norm(P_1 - P_2) / np.linalg.norm(dP_1_unscaled)
        # dP_1 = s_n_1 * dP_1_unscaled

        dP_0_unscaled = (P_1 + P__1) / 2
        dP_0 = self._scale_vector(dP_0_unscaled, P_0, P_1)
        dP_1_unscaled = (P_2 + P_0) / 2
        dP_1 = self._scale_vector(dP_1_unscaled, P_1, P_2)

        # find coefficients
        boundary = np.array([P_0, dP_0, P_1, dP_1])
        A = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [1, 1, 1, 1], [3, 2, 1, 0]])
        coefs = np.linalg.pinv(A) @ boundary
        a = coefs[0, :].reshape(3, 1)
        b = coefs[1, :].reshape(3, 1)
        c = coefs[2, :].reshape(3, 1)
        d = coefs[3, :].reshape(3, 1)

        lambda_list = [0]
        d_lambda_list = [self.prev_d_lambda]

        k = 0
        while (lambda_list[k] < 1):
            p = a * lambda_list[k] ** 3 + b * lambda_list[k] ** 2 + c * lambda_list[k] + d
            dp = 3 * a * lambda_list[k] ** 2 + 2 * b * lambda_list[k] + c

            # D_end = np.linalg.norm(p.reshape(3,1) - P_1.reshape(3,1))
            # vel_current = d_lambda_list[-1] * np.linalg.norm(dp)

            d_lambda_max = _v_max_ / np.linalg.norm(dp)
            dd_lambda_max = _a_max_ / np.linalg.norm(dp)

            # d_lambda_next = 0
            #
            # if D_end <= ((vel_current**2) / 2 * _a_max_):
            #     vel_next = math.sqrt(0.5 * D_end * _a_max_)
            #     d_lambda_next = vel_next / np.linalg.norm(dp)
            # else:


            if d_lambda_list[k] < d_lambda_max:
                d_lambda_next = d_lambda_list[k] + _T_s_ * dd_lambda_max # d_lambda/dt k+1
            else:
                d_lambda_next = d_lambda_max


            lambda_list.append(lambda_list[k] + _T_s_ * d_lambda_next)
            d_lambda_list.append(d_lambda_next)

            self.prev_d_lambda = d_lambda_next
            k += 1

        lambd = np.array(lambda_list)
        d_lambda = np.array(d_lambda_list)

        P = a*lambd**3 + b*lambd**2 + c*lambd + d
        dP = 3 * a * lambd**2 + 2 * b*lambd + c

        v = dP * d_lambda

        # self.prev_s_n = s_n
        return P, v, lambda_list, d_lambda_list




pos_buffer = np.array([[0, 0, 5., 5., 5., 5.],
                       [-10, -10, -5, -5, -5, -5.],
                       [0, 0, 5., 10, 15, 15]])

traj_gen = trajectory_generator()

traject_pos_1, traject_vel_1, lambda_list1, d_lambda_list1 = traj_gen.generate(pos_buffer[:, 0], pos_buffer[:, 1], pos_buffer[:, 2], pos_buffer[:, 3]) # B -> C
traject_pos_2, traject_vel_2, lambda_list2, d_lambda_list2 = traj_gen.generate(pos_buffer[:, 1], pos_buffer[:, 2], pos_buffer[:, 3], pos_buffer[:, 4]) # C -> D
traject_pos_3, traject_vel_3, lambda_list3, d_lambda_list3 = traj_gen.generate(pos_buffer[:, 2], pos_buffer[:, 3], pos_buffer[:, 4], pos_buffer[:, 5]) # D -> E

trajectory = np.concatenate((traject_pos_1, traject_pos_2, traject_pos_3), axis=1)
velocity = np.concatenate((traject_vel_1, traject_vel_2, traject_vel_3), axis=1)

time_vector = np.arange(0, trajectory.shape[1]) * _T_s_

norm_vel = np.linalg.norm(velocity, axis=0)

time_seg_1 = np.ones((300, 1)) * traject_pos_1.shape[1] * _T_s_
time_seg_2 = time_seg_1 + np.ones((300, 1)) * traject_pos_2.shape[1] * _T_s_

fig, (ax1, ax2, norm) = plt.subplots(1, 3)
ax1.set_title('position')
ax1.plot(time_vector, trajectory[0], label='x')
ax1.plot(time_vector, trajectory[1], label='y')
ax1.plot(time_vector, trajectory[2], label='z')
ax1.plot(time_seg_1, np.arange(0,300), lw=1, c='black', linestyle='--', label='Segment 1 -> 2')
ax1.plot(time_seg_2, np.arange(0,300), lw=1, c='grey', linestyle='--', label='Segment 2 -> 3')


time_seg_1 = np.ones((11, 1)) * traject_pos_1.shape[1] * _T_s_
time_seg_2 = time_seg_1 + np.ones((11, 1)) * traject_pos_2.shape[1] * _T_s_

ax2.set_title('velocity')
ax2.plot(time_vector, velocity[0])
ax2.plot(time_vector, velocity[1])
ax2.plot(time_vector, velocity[2])
ax2.plot(time_seg_1, np.arange(0,11), lw=1, c='black', linestyle='--')
ax2.plot(time_seg_2, np.arange(0,11), lw=1, c='grey', linestyle='--')

norm.plot(time_vector, norm_vel, c='red')

fig.legend()
plt.show()
# plt.figure(1)
# plt.ylim([0, .2])
# plt.plot(lambda_list1, d_lambda_list1)
# plt.figure(2)
# plt.ylim([0, .2])
# plt.plot(lambda_list2, d_lambda_list2)
# plt.figure(3)
# plt.ylim([0, .2])
# plt.plot(lambda_list3, d_lambda_list3)
#
# # ax1.legend()
# # ax2.legend()
#
# plt.show()

