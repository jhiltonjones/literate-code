import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import dill as pickle
from scipy.optimize import minimize

# --- Physical parameters ---
L = 0.05                 # total length of catheter
r = 0.00054              # radius
E = 3.0e6                # Young's modulus
A = math.pi * r**2
I = math.pi * r**4 / 4.0
MU0 = 4 * math.pi * 1e-7 # vacuum permeability
M = 8000                 # magnetic material property
MAGNET_M = 318           # magnetization


# Load 3D symbolic terms (assumes full internal dipole in xy-plane, parametrized by theta)
with open('magnetic_field_terms_3D.pkl', 'rb') as f:
    f_first_term_3d, f_second_term_3d = pickle.load(f)

def rot_x(phi):
    return np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi),  np.cos(phi)]
    ])


def compute_dF_dtheta_symbolic_3d(theta_val, x_val, y_val, z_val, magnet_pos, dipole_angles):
    x_m, y_m, z_m = magnet_pos
    psi, phi = dipole_angles
    px, py, pz = x_val - x_m, y_val - y_m, z_val - z_m
    r_mag = np.linalg.norm([px, py, pz])
    if r_mag == 0:
        return 0.0
    C_val = MU0 * MAGNET_M / (4 * np.pi * r_mag**3)

    first_term = f_first_term_3d(theta_val, x_val, y_val, z_val, x_m, y_m, z_m, psi, phi, C_val)
    second_term = f_second_term_3d(theta_val, x_val, y_val, z_val, x_m, y_m, z_m, psi, phi, C_val)
    return first_term + second_term

def solve_deflection_angle_3d(magnet_pos, dipole_angles, n_steps=1000):
    # Initial tangent of the beam
    initial_tangent = np.array([1.0, 0.0, 0.0])  # assuming straight beam along x


    vec = np.array([0.0, 0.0, 0.0]) - np.array(magnet_pos)  # from magnet to catheter base
    vec /= np.linalg.norm(vec)
    psi = np.arccos(vec[2])
    phi = np.arctan2(vec[1], vec[0])
    dipole_angles = (psi, phi)


    def integrate_curvature(k0):
        theta = 0.0
        dtheta = k0
        x, y, z = 0.0, 0.0, 0.0
        ds = L / n_steps
        for _ in range(n_steps):
            roll_angle_phi = np.pi / 4
            t2d = np.array([np.cos(theta), np.sin(theta), 0.0])
            unit_tangent = rot_x(roll_angle_phi) @ t2d

            # Use fixed dipole_angles computed before loop
            ddtheta = -(A * M / (E * I)) * compute_dF_dtheta_symbolic_3d(
                theta, x, y, z, magnet_pos, dipole_angles
            )

            dx, dy, dz = unit_tangent
            theta += dtheta * ds
            dtheta += ddtheta * ds
            x += dx * ds
            y += dy * ds
            z += dz * ds

        return dtheta

    # Secant root solve
    k0_low, k0_high = 0.0, 50.0
    res_low = integrate_curvature(k0_low)
    res_high = integrate_curvature(k0_high)
    attempts = 0
    while res_low * res_high > 0 and attempts < 50:
        k0_low = k0_high
        res_low = res_high
        k0_high *= 2.0
        res_high = integrate_curvature(k0_high)
        attempts += 1

    k0_a, k0_b = k0_low, k0_high
    res_a, res_b = res_low, res_high
    for _ in range(50):
        if abs(res_b - res_a) < 1e-9:
            break
        k0_mid = k0_b - res_b * ((k0_b - k0_a) / (res_b - res_a))
        res_mid = integrate_curvature(k0_mid)
        if abs(res_mid) < 1e-6:
            k0_solution = k0_mid
            break
        if res_a * res_mid < 0:
            k0_b, res_b = k0_mid, res_mid
        else:
            k0_a, res_a = k0_mid, res_mid
    else:
        k0_solution = k0_mid

    # Final integration
    theta = 0.0
    dtheta = k0_solution
    x, y, z = 0.0, 0.0, 0.0
    ds = L / n_steps
    x_vals, y_vals, z_vals, theta_vals = [x], [y], [z], [theta]
    for _ in range(n_steps):
        roll_angle_phi = np.pi / 4
        t2d = np.array([np.cos(theta), np.sin(theta), 0.0])
        unit_tangent = rot_x(roll_angle_phi) @ t2d

        # Use fixed dipole_angles computed before loop
        ddtheta = -(A * M / (E * I)) * compute_dF_dtheta_symbolic_3d(
            theta, x, y, z, magnet_pos, dipole_angles
        )

        dx, dy, dz = unit_tangent
        theta += dtheta * ds
        dtheta += ddtheta * ds
        x += dx * ds
        y += dy * ds
        z += dz * ds

        x_vals.append(x)
        y_vals.append(y)
        z_vals.append(z)
        theta_vals.append(theta)
    s_vals = [i * ds for i in range(len(theta_vals))]
    return s_vals, theta_vals, x_vals, y_vals, z_vals


if __name__ == "__main__":
    magnet_pos = [0.25, -0.01, 0.00]

    s_vals, theta_vals, x_vals, y_vals, z_vals = solve_deflection_angle_3d(magnet_pos, None)

    # Prepare 3D positions for plotting
    pos_vals = np.stack([x_vals, y_vals, z_vals], axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pos_vals[:, 0], pos_vals[:, 1], pos_vals[:, 2], label="Catheter", linewidth=2)
    ax.scatter(*magnet_pos, color='red', s=60, label='Magnet', marker='o')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('External Dipole Aligned with Catheter')
    ax.legend()
    plt.show()

