import numpy as np
import matplotlib.pyplot as plt
def compute_desired_trajectory(k, N, step, Ts=1.0):

    straight_length = 0.03
    arc1_angle_deg = 60
    arc2_angle_deg = -90  
    arc1_length = 0.02
    arc2_length = 0.03

    x1 = np.arange(0.03, 0.03 + straight_length + step, step)
    y1 = np.zeros_like(x1)

    arc1_angle_rad = np.deg2rad(arc1_angle_deg)
    R1 = arc1_length / arc1_angle_rad
    theta1 = np.linspace(0, arc1_angle_rad, int(np.ceil(arc1_length / step)))
    x2 = x1[-1] + R1 * np.sin(theta1)
    y2 = y1[-1] + R1 * (1 - np.cos(theta1))

    arc2_angle_rad = np.deg2rad(abs(arc2_angle_deg))
    R2 = arc2_length / arc2_angle_rad
    theta2 = np.linspace(0, arc2_angle_rad, int(np.ceil(arc2_length / step)))

    dx = R2 * np.sin(theta2)
    dy = R2 * (1 - np.cos(theta2))
    rotation_matrix = np.array([
        [np.cos(arc1_angle_rad), -np.sin(arc1_angle_rad)],
        [np.sin(arc1_angle_rad),  np.cos(arc1_angle_rad)],
    ])
    delta = np.stack((dx, -dy), axis=0)  
    rotated = rotation_matrix @ delta
    x3 = x2[-1] + rotated[0]
    y3 = y2[-1] + rotated[1]

    x_total = np.concatenate([x1, x2, x3])
    y_total = np.concatenate([y1, y2, y3])
    path = np.vstack((x_total, y_total)).T

    idx_start = min(k, path.shape[0] - 1)
    idx_end = min(k + N, path.shape[0])
    Xd = path[idx_start:idx_end]

    while Xd.shape[0] < N:
        Xd = np.vstack([Xd, Xd[-1]])

    diffs = np.diff(path, axis=0)
    total_length = np.sum(np.linalg.norm(diffs, axis=1))
    v_suggested = total_length / (len(path) * Ts)

    return Xd, v_suggested
# def compute_desired_trajectory(k, N, step, Ts=1.0):

#     total_length = 0.08  # total length of the sloped path
#     start_x = 0.03
#     slope = -0.9 # for example, dy/dx = 0.5

#     x_vals = np.arange(start_x, start_x + total_length + step, step)
#     y_vals = slope * (x_vals - start_x)

#     path = np.vstack((x_vals, y_vals)).T

#     idx_start = min(k, path.shape[0] - 1)
#     idx_end = min(k + N, path.shape[0])
#     Xd = path[idx_start:idx_end]

#     while Xd.shape[0] < N:
#         Xd = np.vstack([Xd, Xd[-1]])

#     diffs = np.diff(path, axis=0)
#     total_length = np.sum(np.linalg.norm(diffs, axis=1))
#     v_suggested = total_length / (len(path) * Ts)

#     return Xd, v_suggested




# Xd, v_suggested = compute_desired_trajectory(k=0, N=100, step=0.005)

# plt.figure(figsize=(6, 4))
# plt.plot(Xd[:, 0], Xd[:, 1], 'b-o', label='Desired Trajectory')
# plt.xlabel('x (m)')
# plt.ylabel('y (m)')
# plt.title('Desired Catheter Tip Trajectory')
# plt.axis('equal')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

