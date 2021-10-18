import numpy as np
from math import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.polynomial.polynomial as poly

def import_array_file(file):
    try:
        data = np.genfromtxt(file, delimiter=',')
    except:
        print("Error in opening or reading from file")
        return
    return data


def plot_3d_trajectory(trajectories: list[np.ndarray], colors: list[str], plot_title, blocking=False):
    fig = plt.figure(figsize=(10, 10))
    ax = Axes3D(fig)

    for idx, trajectory in enumerate(trajectories):
        x_vector = trajectory[0, :]
        y_vector = trajectory[1, :]
        z_vector = trajectory[2, :]

        color = colors[idx]
        ax.plot(x_vector, y_vector, z_vector, lw=2, c=color, linestyle='-', label=idx)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    ax.legend()
    fig.suptitle(plot_title)
    plt.show(block=blocking)


def filter_trajectory_poly(trajectory, degree=3):
    trajectory_pos = trajectory[0:3].T
    traject_x = trajectory_pos[:, 0]
    traject_y = trajectory_pos[:, 1]
    traject_z = trajectory_pos[:, 2]

    trajectory_size = int(trajectory_pos.shape[0])
    discrete_vector = np.arange(0, trajectory_size)

    coefs_x = poly.polyfit(discrete_vector, traject_x, degree)
    coefs_y = poly.polyfit(discrete_vector, traject_y, degree)
    coefs_z = poly.polyfit(discrete_vector, traject_z, degree)

    x_filtered = poly.polyval(discrete_vector, coefs_x)
    y_filtered = poly.polyval(discrete_vector, coefs_y)
    z_filtered = poly.polyval(discrete_vector, coefs_z)

    trajectory_pos_filtered = np.array([x_filtered, y_filtered, z_filtered])
    trajectory_filtered = np.vstack((trajectory_pos_filtered, trajectory[3:6]))

    return trajectory_filtered


def calculate_trajectory_start_vector(trajectory, number_of_points):
    sub_sample_trajectory = trajectory[0:3, 0:number_of_points]

    line = filter_trajectory_poly(sub_sample_trajectory, degree=1)
    start = line[:, 0]
    end = line[:, -1]

    return start, end


def skew(vector):
    return np.array([[0, -vector[2][0], vector[1][0]],
                     [vector[2][0], 0, -vector[0][0]],
                     [-vector[1][0], vector[0][0], 0]])

def calculate_rotation_matrix(vector, angle):
    angle = radians(angle)
    vector = vector / np.linalg.norm(vector)
    vector = vector.reshape(3, 1)
    return np.eye(3) + sin(angle) * skew(vector) + (2*sin(angle/2)**2) * np.linalg.matrix_power(skew(vector), 2)


def rotate_trajectory_around_start(trajectory, angle):
    start, end = calculate_trajectory_start_vector(trajectory, 500)

    rotation_matrix = calculate_rotation_matrix(end-start, angle)

    offset_matrix = np.ones(trajectory.shape) * start.reshape(3,1)
    trajectory_to_origin = trajectory - offset_matrix

    trajectory_rotated = (rotation_matrix @ trajectory_to_origin) + offset_matrix
    return trajectory_rotated

if __name__ == '__main__':
    tube_trajectory_up = import_array_file('../Tube_Orientation_Plotting/targetCollector_up.csv')[0:3]
    tube_trajectory_down = import_array_file('../Tube_Orientation_Plotting/targetCollector_down.csv')[0:3]
    tube_trajectory_left = import_array_file('../Tube_Orientation_Plotting/targetCollector_left.csv')[0:3]

    tube_trajectory_up_rotated = rotate_trajectory_around_start(tube_trajectory_up, 165)

    plot_3d_trajectory([tube_trajectory_up, tube_trajectory_down, tube_trajectory_left, tube_trajectory_up_rotated], ['red', 'green', 'blue', 'black'],
                       'Tube Positions', blocking=False)


    tube_trajectory_up_filtered = filter_trajectory_poly(tube_trajectory_up, 30)
    plot_3d_trajectory([tube_trajectory_up, tube_trajectory_up_filtered], ['red', 'black'], 'UP', blocking=True)

