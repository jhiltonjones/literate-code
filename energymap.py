import math
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

output_folder = "figures_output60.2_attract18"
os.makedirs(output_folder, exist_ok=True)

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

MAGNET_M = 346           # magnet's dipole moment magnitude (A·m^2), calibrated for the N52 magnet

magnet_position = np.array([0.02, 0.16])
theta_range_small = np.linspace(-180, 180, 50)
theta1_grid_small, theta2_grid_small = np.meshgrid(theta_range_small, theta_range_small)

# Initialize maps
magnetic_energy_map = np.zeros_like(theta1_grid_small)
elastic_energy_map = np.zeros_like(theta1_grid_small)
bending_angle_map = np.zeros_like(theta1_grid_small)
snapping_metric = np.zeros_like(theta1_grid_small)


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
# --- Step 3: Updated deflection solver using full symbolic dF/dtheta ---
def solve_deflection_angle_energy(magnet_pos, magnet_dipole_angle, n_steps = 1000):
    mag_energy_along_beam = []
    elas_energy_along_beam = []
    dF_dtheta_vals = []
    def integrate_curvature(k0):
        theta = 0.0
        dtheta = k0
        x, y = 0.0, 0.0
        n_steps = 1000
        ds = L / n_steps
        for _ in range(n_steps):
            # if _ < n_steps // 2:
            #     local_dipole_angle = magnet_dipole_angle  
            # else:
            #     local_dipole_angle = magnet_dipole_angle + np.pi # -x direction
            if _ < n_steps // 4:
                local_dipole_angle = magnet_dipole_angle   # No magnetization
            elif _ < 2 * n_steps // 4:
                local_dipole_angle = magnet_dipole_angle  # +x direction
            elif _ < 3 * n_steps // 4:
                local_dipole_angle = magnet_dipole_angle  # +x direction
            else:
                local_dipole_angle = magnet_dipole_angle # -x direction
            
            if local_dipole_angle is None:
                ddtheta = 0.0  # No torque applied
            else:
                ddtheta = -(A * M / (E * I)) * compute_dF_dtheta_symbolic(theta, x, y, magnet_pos, local_dipole_angle)

            dx = np.cos(theta)
            dy = np.sin(theta)
            theta += dtheta * ds
            dtheta += ddtheta * ds
            x += dx * ds
            y += dy * ds

        return dtheta 

    # Bracket the root for θ'(L)=0 by trying different initial curvatures
    k0_low, k0_high = 0.0, 50.0
    res_low = integrate_curvature(k0_low)    # θ'(L) with k0_low
    res_high = integrate_curvature(k0_high)  # θ'(L) with k0_high
    # Expand bracket until sign change
    attempts = 0
    while res_low * res_high > 0:
        k0_low = k0_high
        res_low = res_high
        k0_high *= 2.0
        res_high = integrate_curvature(k0_high)
        attempts += 1
        if attempts > 100:
            print("[WARNING] No sign change in integrate_curvature. Switching to fallback k0 range.")
            # Try the opposite direction
            k0_low, k0_high = -50, 50
            res_low = integrate_curvature(k0_low)
            res_high = integrate_curvature(k0_high)
            break
    # Use secant method to find root k0 such that θ'(L) ≈ 0
    k0_a, k0_b = k0_low, k0_high
    res_a, res_b = res_low, res_high
    k0_mid = (k0_a + k0_b) / 2  # Initialize in case loop doesn't assign it
    k0_solution = None
    for _ in range(50):  # iterate to refine root
        if abs(res_b - res_a) < 1e-9:
            break
        # Secant update for k0
        k0_mid = k0_b - res_b * ((k0_b - k0_a) / (res_b - res_a))
        res_mid = integrate_curvature(k0_mid)
        if abs(res_mid) < 1e-6:  # convergence when θ'(L) is near zero
            k0_solution = k0_mid
            break
        # Update bracketing interval
        if res_a * res_mid < 0:
            k0_b, res_b = k0_mid, res_mid
        else:
            k0_a, res_a = k0_mid, res_mid
        k0_solution = k0_mid
    if k0_solution is None:
        k0_solution = k0_mid  # fallback to last mid value if not converged

    # Integrate one more time with k0_solution to obtain the full θ(s), x(s), y(s) profiles
    theta = 0.0
    dtheta = k0_solution
    x = 0.0;  y = 0.0
    n_steps = 1000
    ds = L / n_steps
    # theta_vals = [theta]
    # x_vals = [x]
    # y_vals = [y]
    theta_vals = []
    x_vals = []
    y_vals = []

    for i in range(n_steps):
        # if _ < n_steps // 2:
        #     local_dipole_angle = magnet_dipole_angle 
        # else:
        #     local_dipole_angle = magnet_dipole_angle + np.pi  # -x direction
        if i < n_steps // 4:
            local_dipole_angle = magnet_dipole_angle   # No magnetization
        elif i < 2 * n_steps // 4:
            local_dipole_angle = magnet_dipole_angle   # +x direction
        elif i < 3 * n_steps // 4:
            local_dipole_angle = magnet_dipole_angle # +x direction
        else:
            local_dipole_angle = magnet_dipole_angle # -x direction
        
        if local_dipole_angle is None:
            ddtheta = 0.0  # No torque applied
        else:
            ddtheta = -(A * M / (E * I)) * compute_dF_dtheta_symbolic(theta, x, y, magnet_pos, local_dipole_angle)

        dx = math.cos(theta)
        dy = math.sin(theta)
        theta  += dtheta * ds
        dtheta += ddtheta * ds
        x += dx * ds
        y += dy * ds
        dF_dtheta = compute_dF_dtheta_symbolic(theta, x, y, magnet_pos, local_dipole_angle)
        mag_energy = -M * dF_dtheta * ds
        elas_energy = 0.5 * E * I * dtheta**2 * ds
        # print(f"dtheta is {dtheta}")
        # print(f"Elastic energy is {elas_energy}")
        # print(f"Magnetic energy is {mag_energy}")
        mag_energy_along_beam.append(mag_energy)
        elas_energy_along_beam.append(elas_energy)
        dF_dtheta_vals.append(dF_dtheta)
        theta_vals.append(theta)
        x_vals.append(x)
        y_vals.append(y)
    s_vals = np.linspace(0, L, n_steps)

    # return s_vals, theta_vals, x_vals, y_vals
    return s_vals, theta_vals, x_vals, y_vals, mag_energy_along_beam, elas_energy_along_beam, dF_dtheta_vals
# psi_deg_range = np.linspace(0, 360, 30)
# theta_tip_vals = []
# total_mag_energies = []
# total_elas_energies = []
# for psi_deg in tqdm(psi_deg_range):
#     psi1 = np.deg2rad(psi_deg)
#     s_vals, theta_vals, x_vals, y_vals, mag_energy_along_beam, elas_energy_along_beam = solve_deflection_angle_energy(magnet_position, psi1)
    
#     # Store tip angle
#     theta_tip_vals.append(theta_vals[-1])

#     # Store total energies
#     total_mag_energy = np.sum(mag_energy_along_beam)
#     total_elas_energy = np.sum(elas_energy_along_beam)
#     total_mag_energies.append(total_mag_energy)
#     total_elas_energies.append(total_elas_energy)
# # psi_deg_range = np.linspace(0, 360, 30)
# # theta_tip_vals = []

# # for psi_deg in tqdm(psi_deg_range):
# #     psi1 = np.deg2rad(psi_deg)
# #     s_vals, theta_vals, x_vals, y_vals, mag_energy_along_beam, elas_energy_along_beam = solve_deflection_angle_energy(magnet_position, psi1)
# #     theta_tip_vals.append(theta_vals[-1])
    
    

# # Convert to np.array for plotting
# theta_tip_vals = np.unwrap(np.array(theta_tip_vals))  # unwrap to remove 2π jumps

# # Compute snapping metric as derivative
# dtheta_dpsi = np.gradient(theta_tip_vals, np.deg2rad(psi_deg_range))

# # Plot results
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.plot(psi_deg_range, np.rad2deg(theta_tip_vals))
# plt.xlabel("ψ₁ (deg)")
# plt.ylabel("Tip Angle θ(L) [deg]")
# plt.title("Final Bending Angle vs Magnetization Angle")

# plt.subplot(1, 2, 2)
# plt.plot(psi_deg_range, dtheta_dpsi)
# plt.xlabel("ψ₁ (deg)")
# plt.ylabel("dθ/dψ₁")
# plt.title("Snapping Sensitivity (Derivative)")

# plt.tight_layout()
# plt.show()

# s_vals = np.linspace(0, L, len(mag_energy_along_beam))
# plt.figure(figsize=(10, 4))
# plt.plot(s_vals, mag_energy_along_beam, label='Magnetic Energy Density')
# plt.plot(s_vals, elas_energy_along_beam, label='Elastic Energy Density')
# plt.xlabel('Beam Arc Length s (m)')
# plt.ylabel('Energy (J)')
# plt.title('Energy Distribution Along Beam (ψ₁ = 60°)')
# plt.legend()
# plt.grid(True)
# plt.show()
# plt.figure(figsize=(10, 4))
# plt.plot(psi_deg_range, total_mag_energies, label='Total Magnetic Energy')
# plt.plot(psi_deg_range, total_elas_energies, label='Total Elastic Energy')
# plt.xlabel('ψ₁ (deg)')
# plt.ylabel('Total Energy (J)')
# plt.title('Total Energy vs Magnetization Angle ψ₁')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# Select a single psi angle (e.g., 60 degrees)
# psi_deg_single = 90
# psi1 = np.deg2rad(psi_deg_single)

# # Solve for this configuration
# s_vals, theta_vals, x_vals, y_vals, mag_energy_along_beam, elas_energy_along_beam, dF_dtheta_vals = solve_deflection_angle_energy(magnet_position, psi1)

# # Compute total energy along the beam
# total_energy_along_beam = np.array(mag_energy_along_beam) + np.array(elas_energy_along_beam)

# # Arc length for plotting
# # s_vals = np.linspace(0, L, len(mag_energy_along_beam))
# # Compute curvature: dtheta/ds
# theta_vals = np.array(theta_vals)
# curvature_vals = np.gradient(theta_vals, s_vals)  # dθ/ds
# dF_dtheta_vals = np.array(dF_dtheta_vals)
# torque_density = M * dF_dtheta_vals

# # Plot: Energy density along the beam
# plt.figure(figsize=(12, 5))
# plt.plot(s_vals, mag_energy_along_beam, label='Magnetic Energy Density', linewidth=2)
# plt.plot(s_vals, elas_energy_along_beam, label='Elastic Energy Density', linewidth=2)
# plt.xlabel('Beam Arc Length s (m)')
# plt.ylabel('Energy Density (J)')
# plt.title(f'Energy Distribution Along Beam (ψ₁ = {psi_deg_single}°)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# # Plot curvature
# plt.figure(figsize=(8, 4))
# plt.plot(s_vals, curvature_vals, label='Curvature (dθ/ds)', color='purple')
# plt.xlabel('Arc Length s (m)')
# plt.ylabel('Curvature κ (1/m)')
# plt.title(f'Curvature Profile Along Beam (ψ₁ = {psi_deg_single}°)')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()


# plt.figure(figsize=(8, 4))
# plt.plot(s_vals, torque_density, label='Magnetic Torque Density M·(dF/dθ)', color='darkgreen')
# plt.xlabel('Arc Length s (m)')
# plt.ylabel('Torque Density (Nm/m)')
# plt.title(f'Magnetic Torque Density Along Beam (ψ₁ = {psi_deg_single}°)')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# # Plot: Total potential energy density
# plt.figure(figsize=(8, 4))
# plt.plot(s_vals, total_energy_along_beam, label='Total Potential Energy Density', color='black', linewidth=2)
# plt.xlabel('Beam Arc Length s (m)')
# plt.ylabel('Total Energy Density (J)')
# plt.title(f'Total Potential Energy Along Beam (ψ₁ = {psi_deg_single}°)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# plt.figure(figsize=(6, 6))

# # Plot catheter deformation
# plt.plot(x_vals, y_vals, linewidth=2, label=f'ψ = {math.degrees(psi_deg_single):.1f}°')

# # Plot magnet position
# plt.scatter(magnet_position[0], magnet_position[1], color='red', s=80, label='Magnet')

# # Add label with coordinates
# plt.text(
#     magnet_position[0] + 0.002,  # slight x offset to avoid overlap
#     magnet_position[1] + 0.002,  # slight y offset
#     f'Magnet\n({magnet_position[0]:.3f}, {magnet_position[1]:.3f})',
#     color='red',
#     fontsize=9,
#     va='bottom',
#     ha='left'
# )

# # Axes and styling
# plt.xlabel('x (m)')
# plt.ylabel('y (m)')
# plt.title('MSCR Deformation Under Magnetic Actuation')
# plt.axis('equal')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

psi_range = np.arange(0, 361, 30)
energy_differences = []

fig_energy, ax_energy = plt.subplots(figsize=(10, 4))
fig_curvature, ax_curv = plt.subplots(figsize=(8, 4))
fig_torque, ax_torque = plt.subplots(figsize=(8, 4))
fig_total_energy, ax_total_energy = plt.subplots(figsize=(8, 4))
fig_deformation, ax_deform = plt.subplots(figsize=(6, 6))
fig_energy_diff, ax_energy_diff = plt.subplots(figsize=(8, 4))

for psi_deg_single in psi_range:
    psi1 = np.deg2rad(psi_deg_single)
    print(f"PSI: {psi_deg_single}")
    # Solve beam deflection
    s_vals, theta_vals, x_vals, y_vals, mag_energy_along_beam, elas_energy_along_beam, dF_dtheta_vals = solve_deflection_angle_energy(magnet_position, psi1)

    theta_vals = np.array(theta_vals)
    curvature_vals = np.gradient(theta_vals, s_vals)
    dF_dtheta_vals = np.array(dF_dtheta_vals)
    torque_density = M * dF_dtheta_vals
    total_energy_along_beam = np.array(mag_energy_along_beam) + np.array(elas_energy_along_beam)
    half = len(total_energy_along_beam) // 2
    first_half_energy = np.sum(total_energy_along_beam[:half])
    second_half_energy = np.sum(total_energy_along_beam[half:])
    diff = np.abs(first_half_energy - second_half_energy)
    energy_differences.append(diff)
    # ENERGY
    ax_energy.plot(s_vals, mag_energy_along_beam)
    ax_energy.plot(s_vals, elas_energy_along_beam, linestyle='--')
    ax_energy.text(s_vals[-1], mag_energy_along_beam[-1] + 0.005, f'Mag ψ={psi_deg_single}°', fontsize=8)
    ax_energy.text(s_vals[-1], elas_energy_along_beam[-1] - 0.005, f'Elas ψ={psi_deg_single}°', fontsize=8)

    # CURVATURE
    ax_curv.plot(s_vals, curvature_vals)
    ax_curv.text(s_vals[-1], curvature_vals[-1], f'ψ={psi_deg_single}°', fontsize=8, va='bottom')

    # TORQUE
    ax_torque.plot(s_vals, torque_density)
    ax_torque.text(s_vals[-1], torque_density[-1], f'ψ={psi_deg_single}°', fontsize=8, va='bottom')

    # TOTAL ENERGY
    ax_total_energy.plot(s_vals, total_energy_along_beam)
    ax_total_energy.text(s_vals[-1], total_energy_along_beam[-1], f'ψ={psi_deg_single}°', fontsize=8, va='bottom')

    # DEFORMATION
    ax_deform.plot(x_vals, y_vals, linewidth=2)
    ax_deform.text(x_vals[-1], y_vals[-1], f'{psi_deg_single}°', fontsize=8, va='bottom', ha='left')

    
ax_energy_diff.plot(psi_range, energy_differences, marker='o', linestyle='-', color='darkorange')  

# Magnet label on deformation plot
ax_deform.scatter(magnet_position[0], magnet_position[1], color='red', s=80, label='Magnet')
ax_deform.text(magnet_position[0] + 0.002, magnet_position[1] + 0.002,
               f'Magnet\n({magnet_position[0]:.3f}, {magnet_position[1]:.3f})',
               color='red', fontsize=9, va='bottom', ha='left')

for ax, title, xlabel, ylabel, fname in [
    (ax_energy, 'Energy Densities vs Arc Length', 's (m)', 'Energy (J)', "combined_energy_densities.png"),
    (ax_curv, 'Curvature vs Arc Length', 's (m)', 'Curvature κ (1/m)', "combined_curvature.png"),
    (ax_torque, 'Torque Density vs Arc Length', 's (m)', 'Torque Density (Nm/m)', "combined_torque_density.png"),
    (ax_total_energy, 'Total Energy vs Arc Length', 's (m)', 'Total Energy (J)', "combined_total_energy.png"),
    (ax_deform, 'MSCR Deformation Curves', 'x (m)', 'y (m)', "combined_deformations.png"),
    (ax_energy_diff, 'Energy Asymmetry Along Beam vs Magnetization Angle', 'ψ₁ (deg)', 'ΔEnergy (J)', "energy_asymmetry_vs_psi.png")
]:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    if ax == ax_deform:
        ax.axis('equal')
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, fname), dpi=300)


theta_tip_vals = []
total_mag_energies = []
total_elas_energies = []

# Run loop and collect data
for psi_deg in tqdm(psi_range):
    psi1 = np.deg2rad(psi_deg)
    s_vals, theta_vals, x_vals, y_vals, mag_energy_along_beam, elas_energy_along_beam, _ = solve_deflection_angle_energy(magnet_position, psi1)

    theta_tip_vals.append(theta_vals[-1])
    total_mag_energies.append(np.sum(mag_energy_along_beam))
    total_elas_energies.append(np.sum(elas_energy_along_beam))

# Unwrap and compute snapping sensitivity
theta_tip_vals = np.unwrap(np.array(theta_tip_vals))
dtheta_dpsi = np.gradient(theta_tip_vals, np.deg2rad(psi_range))

# Plotting results
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Final Bending Angle
axs[0, 0].plot(psi_range, np.rad2deg(theta_tip_vals))
axs[0, 0].set_xlabel("ψ₁ (deg)")
axs[0, 0].set_ylabel("Tip Angle θ(L) [deg]")
axs[0, 0].set_title("Final Bending Angle vs Magnetization Angle")
axs[0, 0].grid(True)

# Snapping Sensitivity
axs[0, 1].plot(psi_range, dtheta_dpsi)
axs[0, 1].set_xlabel("ψ₁ (deg)")
axs[0, 1].set_ylabel("dθ/dψ₁")
axs[0, 1].set_title("Snapping Sensitivity (Derivative)")
axs[0, 1].grid(True)

# Total Magnetic Energy
axs[1, 0].plot(psi_range, total_mag_energies, label='Magnetic')
axs[1, 0].plot(psi_range, total_elas_energies, label='Elastic', linestyle='--')
axs[1, 0].set_xlabel("ψ₁ (deg)")
axs[1, 0].set_ylabel("Total Energy (J)")
axs[1, 0].set_title("Total Energy vs Magnetization Angle ψ₁")
axs[1, 0].legend()
axs[1, 0].grid(True)

# Placeholder arc-wise energy for visual consistency
axs[1, 1].plot(s_vals, mag_energy_along_beam, label='Magnetic Energy Density')
axs[1, 1].plot(s_vals, elas_energy_along_beam, label='Elastic Energy Density', linestyle='--')
axs[1, 1].set_xlabel("Beam Arc Length s (m)")
axs[1, 1].set_ylabel("Energy (J)")
axs[1, 1].set_title("Energy Distribution Along Beam (ψ₁ = {:.0f}°)".format(psi_range[5]))
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, "summary_analysis_plots.png"), dpi=300)

plt.show()