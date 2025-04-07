import numpy as np
import matplotlib.pyplot as plt

# Physical constants and magnet parameters
mu0 = 4 * np.pi * 1e-7  # Vacuum permeability (T·m/A)
Br = 1.31               # Remanence (T)
r = 0.04               # EPM radius (m)
h = 0.09                # EPM height (m)
r_i = 0.0005            # IPM radius (m)
h_i = 0.006             # IPM height (m)

# Dipole magnitudes
V_E = np.pi * r**2 * h
V_I = np.pi * r_i**2 * h_i
m_E_mag = (Br * V_E) / mu0
m_I_mag = (Br * V_I) / mu0

# IPM dipole fixed along x
m_I = np.array([m_I_mag, 0, 0])
m_I_hat = m_I / np.linalg.norm(m_I)

# Simulation parameters
angles_rad = np.linspace(0, 6, 50)           # Rotation angles in radians (0 to ~343 deg)
distances = np.linspace(-0.05, 0.05, 50)     # Distance from -5 cm to +5 cm

force_magnitudes = np.zeros((len(distances), len(angles_rad)))

# Compute force magnitude and cap it at 5N
for i, d in enumerate(distances):
    for j, theta in enumerate(angles_rad):
        m_E = m_E_mag * np.array([np.sin(theta), np.cos(theta), 0])
        m_E_hat = m_E / np.linalg.norm(m_E)

        p = np.array([0, d, 0])
        p_norm = np.linalg.norm(p)
        if p_norm == 0:
            force_magnitudes[i, j] = 0
            continue
        p_hat = p / p_norm

        Z = np.eye(3) - 5 * np.outer(p_hat, p_hat)
        D = np.eye(3) - 3 * np.outer(p_hat, p_hat)

        lambda_ = (mu0 / (4 * np.pi)) * np.linalg.norm(m_E) * np.linalg.norm(m_I) / (p_norm**3)

        term1 = np.outer(m_E_hat, m_I_hat)
        term2 = np.outer(m_I_hat, m_E_hat)
        term3 = (m_I_hat @ Z @ m_I_hat) * np.eye(3)
        F_m = (3 * lambda_ / p_norm) * (term1 + term2 + term3) @ p

        force_magnitude = np.linalg.norm(F_m)
        force_magnitudes[i, j] = min(force_magnitude, 5.0)  # Cap at 5 N

# Plotting capped 3D surface
X, Y = np.meshgrid(angles_rad, distances)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, force_magnitudes, cmap='plasma')
ax.set_xlabel('EPM Rotation (radians)')
ax.set_ylabel('Distance (m)')
ax.set_zlabel('Force Magnitude (N)')
ax.set_title('Magnetic Force (Capped at 5 N) vs. EPM Rotation and Distance')
plt.tight_layout()

plt.show()
# Compute the torque magnitude without capping
# Compute the Z-component of the torque and cap it at ±0.1 Nm
torque_z_capped = np.zeros((len(distances), len(angles_rad)))

for i, d in enumerate(distances):
    for j, theta in enumerate(angles_rad):
        m_E = m_E_mag * np.array([np.sin(theta), np.cos(theta), 0])
        m_E_hat = m_E / np.linalg.norm(m_E)

        p = np.array([0, d, 0])
        p_norm = np.linalg.norm(p)
        if p_norm == 0:
            torque_z_capped[i, j] = 0
            continue
        p_hat = p / p_norm

        Z = np.eye(3) - 5 * np.outer(p_hat, p_hat)
        D = np.eye(3) - 3 * np.outer(p_hat, p_hat)

        lambda_ = (mu0 / (4 * np.pi)) * np.linalg.norm(m_E) * np.linalg.norm(m_I) / (p_norm**3)

        T_m = np.cross(lambda_ * m_I_hat, D @ m_E_hat)
        T_z = T_m[2]

        # Cap T_z to ±0.1 Nm
        torque_z_capped[i, j] = np.clip(T_z, -0.1, 0.1)

# Plotting the capped Z-torque component
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, torque_z_capped, cmap='seismic')
ax.set_xlabel('EPM Rotation (radians)')
ax.set_ylabel('Distance (m)')
ax.set_zlabel('Torque Z Component (Nm)')
ax.set_title('Capped Z-Component of Magnetic Torque (±0.1 Nm)')
plt.tight_layout()

plt.show()

