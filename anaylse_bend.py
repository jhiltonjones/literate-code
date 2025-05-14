import numpy as np
import matplotlib.pyplot as plt
from force_andtorque import force_from_paper

# === Parameters ===
magnet_distance = 0.165         # Magnet is 20cm above tip
steps = 50                    # Steps to simulate (though fixed tip here)

# === Initial tip position ===
tip_initial = np.array([0.03, 0.0])

# === Sweep magnet rotation angles ===
theta_vals = np.linspace(0, 180, 50)  # Fine sweep
deflection_angles = []
torque_z_values = []
bending_angles = []

for theta in theta_vals:
    # Magnet is fixed above the initial tip (always same position)
    magnet_pos = np.array([tip_initial[0], tip_initial[1] + magnet_distance, 0.0])

    # r_vec: magnet relative to tip
    r_vec = magnet_pos - np.array([tip_initial[0], tip_initial[1], 0.0])

    # Compute magnetic force and torque
    F_m, T_m = force_from_paper(r_vec, theta)
    torque_z = -T_m[2]  # Out-of-plane torque (z)
    T_m = -T_m[2]
    F_m = F_m[0]
    # Compute bending purely from torque (ignore catheter deformation)
    Ev = 3e6
    Iv = 4.1e-13
    L = 0.03
    gamma_c_hat = (F_m * L**3) / (3 * Ev * Iv) + (T_m * L**2) / (2 * Ev * Iv)
    theta_c_hat = (F_m * L**2) / (2 * Ev * Iv) + (T_m * L) / (Ev * Iv)
    vertical_deflection_deg = gamma_c_hat # small angle approx
    bending_deg = np.rad2deg(theta_c_hat)
    deflection_angles.append(vertical_deflection_deg)
    torque_z_values.append(torque_z)
    bending_angles.append(bending_deg)

    print(f"Magnet Rotation {theta:.2f} deg | Verical Deflection {float(vertical_deflection_deg):.6f} mm | Torque_z {torque_z:.3e} Nm | Bending Angle {float(bending_deg):.3e} Degrees")


# === Plot ===
plt.figure(figsize=(12,6))

plt.subplot(3, 1, 1)
plt.plot(theta_vals, deflection_angles, 'o-', label='Pure Bending Angle')
plt.axhline(0, color='black', linestyle='--')
plt.ylabel('Vertical Deflection')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(theta_vals, torque_z_values, 's--', label='Torque_z')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('Magnet Rotation (degrees)')
plt.ylabel('Torque_z (Nm)')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(theta_vals, bending_angles, 's--', label='Bending')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('Magnet Rotation (degrees)')
plt.ylabel('Bending degrees')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()