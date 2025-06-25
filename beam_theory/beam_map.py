import numpy as np

Ev = 3e6               # Young's modulus (Pa)
Iv = 4.1e-13           # Moment of inertia (m^4)
L = 0.05               # Total catheter length (m)


mu0 = 4 * np.pi * 1e-7  # Vacuum permeability
Br = 1.45               # Remanence
r = 0.04                # EPM radius (m)
h = 0.09                # EPM height (m)
r_i = 0.0005            # IPM radius (m)
h_i = 0.005            # IPM height (m)

V_E = np.pi * r**2 * h
V_I = np.pi * r_i**2 * h_i
m_E_mag = (Br * V_E) / mu0
m_I_mag = (Br * V_I) / mu0
m_I = np.array([m_I_mag, 0, 0])
m_I_hat = m_I / np.linalg.norm(m_I)

pos_epm = np.array([0, 0.25, 0])
p = pos_epm
p_norm = np.linalg.norm(p)
p_hat = p / p_norm

Z = np.eye(3) - 5 * np.outer(p_hat, p_hat)
D = np.eye(3) - 3 * np.outer(p_hat, p_hat)

angles_deg = np.linspace(0, 180, 100)
theta_c_hat_vals = []

for angle_deg in angles_deg:
    theta = np.deg2rad(angle_deg)
    m_E = m_E_mag * np.array([np.sin(theta), np.cos(theta), 0])
    m_E_hat = m_E / np.linalg.norm(m_E)

    lambda_ = (mu0 / (4 * np.pi)) * np.linalg.norm(m_E) * np.linalg.norm(m_I) / (p_norm**3)
    F_m = (3 * lambda_ / p_norm) * ((np.outer(m_E_hat, m_I_hat) + np.outer(m_I_hat, m_E_hat)) + (m_I_hat @ Z @ m_I_hat) * np.eye(3)) @ p
    T_m = np.cross(lambda_ * m_I_hat, D @ m_E_hat)

    Fmi = F_m[1]
    Tmi = -T_m[2]

    theta_c_hat = (Fmi * L**2) / (2 * Ev * Iv) + (Tmi * L) / (Ev * Iv)
    theta_c_hat_vals.append(np.rad2deg(theta_c_hat))  

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(angles_deg, theta_c_hat_vals, linewidth=2)
plt.xlabel('EPM Rotation Angle (degrees)')
plt.ylabel('Estimated Bending Angle θ̂_c (degrees)')
plt.title('Analytical Bending Angle vs EPM Rotation')
plt.grid(True)
plt.tight_layout()
plt.show()

x_vals = np.linspace(-0.2, 0.2, 50)
y_vals = np.linspace(0.22, 0.4, 50)  
theta_c_hat_grid = np.zeros((len(y_vals), len(x_vals)))

m_E = m_E_mag * np.array([0, 1, 0])
m_E_hat = m_E / np.linalg.norm(m_E)

for i, y in enumerate(y_vals):
    for j, x in enumerate(x_vals):
        p = np.array([x, y, 0])
        p_norm = np.linalg.norm(p)
        p_hat = p / p_norm

        Z = np.eye(3) - 5 * np.outer(p_hat, p_hat)
        D = np.eye(3) - 3 * np.outer(p_hat, p_hat)

        lambda_ = (mu0 / (4 * np.pi)) * np.linalg.norm(m_E) * np.linalg.norm(m_I) / (p_norm**3)

        F_m = (3 * lambda_ / p_norm) * ((np.outer(m_E_hat, m_I_hat) + np.outer(m_I_hat, m_E_hat)) + (m_I_hat @ Z @ m_I_hat) * np.eye(3)) @ p
        T_m = np.cross(lambda_ * m_I_hat, D @ m_E_hat)

        Fmi = F_m[1]
        Tmi = -T_m[2]

        theta_c_hat = (Fmi * L**2) / (2 * Ev * Iv) + (Tmi * L) / (Ev * Iv)
        theta_c_hat_grid[i, j] = np.rad2deg(theta_c_hat)

X, Y = np.meshgrid(x_vals, y_vals)
theta_c_hat_grid_capped = np.clip(theta_c_hat_grid, None, 140)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Use MathText (default)
plt.rcParams['font.family'] = 'DejaVu Sans'

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, theta_c_hat_grid_capped, cmap='viridis')

# Use math-style strings (no \textbf, just plain math expressions)
ax.set_xlabel(r'EPM X Position (m)', fontsize=13, labelpad=10)
ax.set_ylabel(r'EPM Y Position (m)', fontsize=13, labelpad=10)
ax.set_zlabel(r'Estimated Bending Angle $\hat{\theta}_c$ (degrees)', fontsize=13, labelpad=10)
ax.set_title(r'Bending Angle vs. EPM X-Y Position', fontsize=14, pad=15)

ax.tick_params(labelsize=11)

# Optional colorbar with MathText
fig.colorbar(surf, ax=ax, shrink=0.7, aspect=12, pad=0.1,
             label=r'$\hat{\theta}_c$ (degrees)')

plt.tight_layout()
plt.show()
