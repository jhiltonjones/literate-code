import math
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm
# Parameters for MSCR #1 (length, radius, modulus, magnetization)
L = 0.08  # rod length in meters (24 mm)
r = 0.00054  # rod radius in meters (0.54 mm)
E = 3.0e6   # Young's modulus in Pascals (3.0 MPa)
# M = 8000.0  # magnetization in A/m (8.0 kA/m)
# pos1 = 0.05
# pos2= 0.14
# Cross-sectional area and second moment of inertia for a circular rod
A = math.pi * r**2
I = math.pi * r**4 / 4.0

# Magnetic constants for the external magnet (point dipole model)
MU0 = 4 * math.pi * 1e-7      # vacuum permeability (μ0)
M = 8000

MAGNET_M = 318           # magnet's dipole moment magnitude (A·m^2), calibrated for the N52 magnet

magnet_position = np.array([0.02, 0.15])


def compute_dF_dtheta_symbolic(theta_val, x_val, y_val, magnet_pos, magnet_dipole_angle):
    x_m, y_m = magnet_pos
    px = x_val - x_m
    py = y_val - y_m
    r_sq = px**2 + py**2
    if r_sq == 0:
        return 0.0
    r_mag = np.sqrt(r_sq)

    # Field constant
    C_val = MU0 * MAGNET_M / (4 * np.pi * r_mag**3)
    import dill as pickle

    with open('magnetic_field_terms.pkl', 'rb') as f:
        f_first_term, f_second_term = pickle.load(f)
    # First term (symbolically lambdified)
    first_term = f_first_term(theta_val, x_val, y_val, x_m, y_m, magnet_dipole_angle, C_val)
    # print("First term:", first_term)

    # Manual form computation
    a = px / r_mag
    b = py / r_mag
    dot_pm = a * np.cos(magnet_dipole_angle) + b * np.sin(magnet_dipole_angle)

    Bx = C_val * (3 * dot_pm * a - np.cos(magnet_dipole_angle))
    By = C_val * (3 * dot_pm * b - np.sin(magnet_dipole_angle))

    val2 = -np.sin(theta_val) * Bx + np.cos(theta_val) * By
    # print("Manual value:", val2)

    # Second term (symbolically lambdified)
    second_term = f_second_term(theta_val, x_val, y_val, x_m, y_m, magnet_dipole_angle, C_val)
    # print("Second term:", second_term)
    # fd_second_term = finite_difference_second_term(theta_val, x_val, y_val, x_m, y_m, magnet_dipole_angle)
    # print("Finite-difference second term:", fd_second_term)
    
    # Total derivative

    total_symbolic = first_term + second_term
    # fd_total = total_finite_difference(theta_val, x_val, y_val, x_m, y_m, magnet_dipole_angle)
    # print("Total symbolic derivative:", total_symbolic)
    # print("Total finite difference:", fd_total)
    # print("Absolute error:", abs(total_symbolic - fd_total))

    return total_symbolic
# Reduce the grid size for faster computation
theta_range_small = np.linspace(-180, 180, 50)
theta1_grid_small, theta2_grid_small = np.meshgrid(theta_range_small, theta_range_small)

magnetic_energy_map = np.zeros_like(theta1_grid_small)
elastic_energy_map = np.zeros_like(theta1_grid_small)

for i in tqdm(range(theta1_grid_small.shape[0])):
    for j in range(theta1_grid_small.shape[1]):
        psi1 = np.deg2rad(theta1_grid_small[i, j])
        psi2 = np.deg2rad(theta2_grid_small[i, j])

        def integrate_with_split_magnet(psi1, psi2):
            theta = 0.0
            dtheta = 0.0
            x = y = 0.0
            magnetic_energy = 0.0
            elastic_energy = 0.0
            ds = L / 500
            for k in range(500):
                local_psi = psi1 if k < 250 else psi2
                dF_dtheta = compute_dF_dtheta_symbolic(theta, x, y, magnet_position, local_psi)
                ddtheta = -(A * M / (E * I)) * dF_dtheta

                magnetic_energy += -M * dF_dtheta * ds
                elastic_energy += 0.5 * E * I * dtheta**2 * ds

                theta += dtheta * ds
                dtheta += ddtheta * ds
                x += np.cos(theta) * ds
                y += np.sin(theta) * ds

            return magnetic_energy, elastic_energy

        mag_E, elas_E = integrate_with_split_magnet(psi1, psi2)
        magnetic_energy_map[i, j] = mag_E
        elastic_energy_map[i, j] = elas_E

# Plot updated maps
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.contourf(theta1_grid_small, theta2_grid_small, magnetic_energy_map, levels=50, cmap='Spectral')
plt.colorbar(label='Magnetic Energy (J)')
plt.title("Magnetic Energy Map")
plt.xlabel(r'$\psi_1$ (°)')
plt.ylabel(r'$\psi_2$ (°)')

plt.subplot(1, 2, 2)
plt.contourf(theta1_grid_small, theta2_grid_small, elastic_energy_map, levels=50, cmap='viridis')
plt.colorbar(label='Elastic Energy (J)')
plt.title("Elastic Energy Map")
plt.xlabel(r'$\psi_1$ (°)')
plt.ylabel(r'$\psi_2$ (°)')

plt.tight_layout()
plt.show()
