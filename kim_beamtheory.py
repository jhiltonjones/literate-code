import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Define the key analytical expressions from the paper

def xi_function(phi, theta_L):
    def integrand(theta):
        arg = np.cos(phi - theta_L) - np.cos(phi - theta)
        return 1.0 / np.sqrt(arg) if arg > 0 else 0.0
    value, _ = quad(integrand, 0, theta_L)
    return value

def X_function(phi, theta_L):
    def integrand(theta):
        arg = np.cos(phi - theta_L) - np.cos(phi - theta)
        return np.cos(theta) / np.sqrt(arg) if arg > 0 else 0.0
    value, _ = quad(integrand, 0, theta_L)
    return value

def Y_function(phi, theta_L):
    def integrand(theta):
        arg = np.cos(phi - theta_L) - np.cos(phi - theta)
        return np.sin(theta) / np.sqrt(arg) if arg > 0 else 0.0
    value, _ = quad(integrand, 0, theta_L)
    return value

def magnetic_field_dipole(r_vec, m_vec):
    r_mag = np.linalg.norm(r_vec)
    if r_mag < 1e-8:
        return np.zeros(2)
    r_hat = r_vec / r_mag
    mu0 = 4 * np.pi * 1e-7
    term1 = 3 * np.dot(r_hat, m_vec) * r_hat
    B_vec = (mu0 / (4 * np.pi * r_mag**3)) * (term1 - m_vec)
    return B_vec

# Example parameters
EI = 3e6 * (np.pi * (0.00054)**4 / 4)  # E * I
A = np.pi * (0.00054)**2              # cross-sectional area
L = 0.08                              # length
M_r = 8000                            # remanent magnetization
# Magnetic dipole setup
magnet_position = np.array([0.0, 0.2])  # place magnet above beam origin
z = 0.132
B_target = 0.08
mu0 = 4 * np.pi * 1e-7
magnitude = (2 * np.pi * z**3 * B_target) / mu0
magnet_dipole_angle = np.pi / 4  # same as φ
m_vec = magnitude * np.array([np.cos(magnet_dipole_angle), np.sin(magnet_dipole_angle)])
# Compute constant uniform magnetic field
def uniform_B_from_dipole(magnet_pos, m_vec, eval_point=np.array([0.0, 0.0])):
    r_vec = eval_point - magnet_pos
    r_mag = np.linalg.norm(r_vec)
    if r_mag < 1e-8:
        return np.zeros(2)
    r_hat = r_vec / r_mag
    mu0 = 4 * np.pi * 1e-7
    term1 = 3 * np.dot(r_hat, m_vec) * r_hat
    B_vec = (mu0 / (4 * np.pi * r_mag**3)) * (term1 - m_vec)
    return B_vec

B_vec_uniform = uniform_B_from_dipole(magnet_position, m_vec, eval_point=np.array([0.0, 0.0]))
B_mag = np.linalg.norm(B_vec_uniform)
phi = np.arctan2(B_vec_uniform[1], B_vec_uniform[0])  # angle of B field


# Sweep over θ_L
theta_L_vals = np.linspace(0.01, np.pi/2, 100)
phi = np.pi / 4  # magnetic field angle

# Evaluate ξ and compare with theoretical energy parameter
xi_vals = np.array([xi_function(phi, theta_L) for theta_L in theta_L_vals])
B_vals = np.full_like(theta_L_vals, B_mag)  # constant B
LHS_vals = (M_r * B_vals * A * L**2) / EI

RHS = 0.5 * xi_vals**2

# Evaluate displacement
X_vals = np.array([X_function(phi, theta_L) for theta_L in theta_L_vals])
Y_vals = np.array([Y_function(phi, theta_L) for theta_L in theta_L_vals])

# Scale displacements to actual units
scales = np.sqrt(EI / (2 * M_r * B_vals * A))
delta_x_vals = scales * X_vals
delta_y_vals = scales * Y_vals

from scipy.optimize import root_scalar

def target_theta_L(phi, target_L, EI, M_r, B, A):
    """Solve for θ_L given fixed L using ξ(φ, θ_L)"""
    def objective(theta_L):
        xi_val = xi_function(phi, theta_L)
        predicted_L = np.sqrt(EI / (2 * M_r * B * A)) * xi_val
        return predicted_L - target_L

    # Solve for θ_L in a reasonable range
    sol = root_scalar(objective, bracket=[0.1, np.pi / 2 - 0.01], method='brentq')
    if not sol.converged:
        raise RuntimeError("Failed to solve for θ_L.")
    return sol.root

# Solve for the unique θ_L that gives the beam length L
# Solve for the unique θ_L that gives the beam length L
theta_L = target_theta_L(phi, L, EI, M_r, B_mag, A)

# Now compute displacement
xi_val = xi_function(phi, theta_L)
X_val = X_function(phi, theta_L)
Y_val = Y_function(phi, theta_L)
scale = np.sqrt(EI / (2 * M_r * B_mag * A))
delta_x = scale * X_val
delta_y = scale * Y_val

plt.figure(figsize=(6, 6))
plt.plot([0, delta_x], [0, delta_y], 'k--', label='Beam')
plt.scatter(0, 0, color='black', label='Base')
plt.scatter(magnet_position[0], magnet_position[1], color='red', s=60, label='Magnet')
plt.quiver(magnet_position[0], magnet_position[1], m_vec[0], m_vec[1],
           scale=1 / 0.01, color='red', label='Magnet dipole')
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.axis("equal")
plt.legend()
plt.grid(True)
plt.title("Fixed-Length Beam and External Magnet")
plt.tight_layout()
plt.show()

import pandas as pd


df = pd.DataFrame({
    "theta_L (deg)": np.degrees(theta_L_vals),
    "xi(phi, theta_L)": xi_vals,
    "0.5 * xi^2": RHS,
    "delta_x (m)": delta_x_vals,
    "delta_y (m)": delta_y_vals
})
plt.figure(figsize=(6, 6))

# Plot beam as a line from origin to final tip position
plt.plot([0, delta_x_vals[-1]], [0, delta_y_vals[-1]], 'k--', label='Beam (tip path)')

# Plot beam tip trajectory
plt.plot(delta_x_vals, delta_y_vals, label='Tip trajectory', color='blue')

# Plot magnet as a red arrow
magnet_arrow_scale = 0.01  # scale for visibility
plt.quiver(
    magnet_position[0], magnet_position[1],           # base of arrow
    m_vec[0], m_vec[1],                               # direction of arrow
    scale=1 / magnet_arrow_scale, color='red', label='Magnet dipole'
)

# Annotate beam base and magnet
plt.scatter(0, 0, color='black', label='Beam base')
plt.scatter(magnet_position[0], magnet_position[1], color='red', s=60)

# Axes
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.axis("equal")
plt.title("Magnet and Beam in Space")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# plt.figure(figsize=(6, 4))
# plt.plot(np.degrees(theta_L_vals), delta_x_vals, label='δₓ (x displacement)')
# plt.plot(np.degrees(theta_L_vals), delta_y_vals, label='δᵧ (y displacement)')
# plt.xlabel('Tip Angle θₗ (degrees)')
# plt.ylabel('Displacement (m)')
# plt.title('Beam Tip Displacement vs Tip Angle')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# plt.figure(figsize=(6, 4))
# plt.plot(np.degrees(theta_L_vals), RHS, label='0.5·ξ²(θₗ)')
# plt.plot(np.degrees(theta_L_vals), LHS_vals, 'r--', label='LHS (dipole field)')
# plt.xlabel('Tip Angle θₗ (degrees)')
# plt.ylabel('Energy Balance Value')
# plt.title('Dimensionless Energy Balance Check')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# plt.figure(figsize=(6, 6))
# plt.plot(delta_x_vals, delta_y_vals, label='Tip trajectory')
# plt.xlabel('δₓ (m)')
# plt.ylabel('δᵧ (m)')
# plt.title('Beam Tip Trajectory in Space')
# plt.axis('equal')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()
# plt.figure(figsize=(6, 4))
# plt.plot(np.degrees(theta_L_vals), xi_vals)
# plt.xlabel('Tip Angle θₗ (degrees)')
# plt.ylabel('ξ(φ, θₗ)')
# plt.title('Non-Dimensional Arc-Length Integral')
# plt.grid(True)
# plt.tight_layout()
# plt.show()
