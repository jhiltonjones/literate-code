import numpy as np

# Physical properties
Ev = 3e6               # Young's modulus (Pa)
Iv = 4.1e-13           # Moment of inertia (m^4)
L = 0.08               # Total catheter length (m)

# Magnet parameters
mu0 = 4 * np.pi * 1e-7  # Vacuum permeability
Br = 1.31               # Remanence
r = 0.04                # EPM radius (m)
h = 0.09                # EPM height (m)
r_i = 0.0005            # IPM radius (m)
h_i = 0.01             # IPM height (m)

V_E = np.pi * r**2 * h
V_I = np.pi * r_i**2 * h_i
m_E_mag = (Br * V_E) / mu0
m_I_mag = (Br * V_I) / mu0
m_I = np.array([m_I_mag, 0, 0])
m_I_hat = m_I / np.linalg.norm(m_I)

# EPM position
pos_epm = np.array([0, 0.35, 0])
p = pos_epm
p_norm = np.linalg.norm(p)
p_hat = p / p_norm

# Precompute Z and D matrices
Z = np.eye(3) - 5 * np.outer(p_hat, p_hat)
D = np.eye(3) - 3 * np.outer(p_hat, p_hat)

# Analytical bending angle for varying EPM dipole angle
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

    # Equation (5) for analytical bending angle
    theta_c_hat = (Fmi * L**2) / (2 * Ev * Iv) + (Tmi * L) / (Ev * Iv)
    theta_c_hat_vals.append(np.rad2deg(theta_c_hat))  # Convert to degrees

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(angles_deg, theta_c_hat_vals, linewidth=2)
plt.xlabel('EPM Rotation Angle (degrees)')
plt.ylabel('Estimated Bending Angle θ̂_c (degrees)')
plt.title('Analytical Bending Angle vs EPM Rotation')
plt.grid(True)
plt.tight_layout()
plt.show()
# Create a grid of x and y positions for the EPM
x_vals = np.linspace(-0.2, 0.2, 50)
y_vals = np.linspace(0.3, 0.4, 50)  # Avoid 0 to prevent division by zero
theta_c_hat_grid = np.zeros((len(y_vals), len(x_vals)))

# EPM dipole fixed at 0 degrees (aligned along +y)
m_E = m_E_mag * np.array([0, 1, 0])
m_E_hat = m_E / np.linalg.norm(m_E)

# Loop over position grid
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

        # Equation (5) for analytical bending angle
        theta_c_hat = (Fmi * L**2) / (2 * Ev * Iv) + (Tmi * L) / (Ev * Iv)
        theta_c_hat_grid[i, j] = np.rad2deg(theta_c_hat)

# Plotting bending angle as a surface over (x, y) positions
X, Y = np.meshgrid(x_vals, y_vals)
# Cap the bending angle grid at 140 degrees
theta_c_hat_grid_capped = np.clip(theta_c_hat_grid, None, 140)

# Plotting the capped bending angle as a surface
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, theta_c_hat_grid_capped, cmap='viridis')
ax.set_xlabel('EPM X Position (m)')
ax.set_ylabel('EPM Y Position (m)')
ax.set_zlabel('Estimated Bending Angle θ̂_c (deg)')
ax.set_title('Bending Angle vs. EPM X and Y Position (Capped at 140°)')
plt.tight_layout()
plt.show()

