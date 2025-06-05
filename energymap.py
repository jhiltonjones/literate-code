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

magnet_position = np.array([0.02, 0.2])
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
                local_dipole_angle = magnet_dipole_angle  # No magnetization
            elif _ < 2 * n_steps // 4:
                local_dipole_angle = magnet_dipole_angle   # +x direction
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
    theta_vals = [theta]
    x_vals = [x]
    y_vals = [y]
    for i in range(n_steps):
        # if _ < n_steps // 2:
        #     local_dipole_angle = magnet_dipole_angle 
        # else:
        #     local_dipole_angle = magnet_dipole_angle + np.pi  # -x direction
        if _ < n_steps // 4:
            local_dipole_angle = magnet_dipole_angle # No magnetization
        elif _ < 2 * n_steps // 4:
            local_dipole_angle = magnet_dipole_angle# +x direction
        elif _ < 3 * n_steps // 4:
            local_dipole_angle = magnet_dipole_angle   # +x direction
        else:
            local_dipole_angle = magnet_dipole_angle  # -x direction
        
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
        theta_vals.append(theta)
        x_vals.append(x)
        y_vals.append(y)
    s_vals = [i * ds for i in range(len(theta_vals))]
    return s_vals, theta_vals, x_vals, y_vals


# Main grid loop
psi_deg_range = np.linspace(0, 360, 30)
theta_tip_vals = []

for psi_deg in tqdm(psi_deg_range):
    # In main loop:
    psi1 = np.deg2rad(psi_deg)
    s_vals, theta_vals, x_vals, y_vals = solve_deflection_angle_energy(magnet_position, psi1)
    theta_tip = theta_vals[-1]
    theta_tip_vals.append(theta_tip)

# Convert to np.array for plotting
theta_tip_vals = np.unwrap(np.array(theta_tip_vals))  # unwrap to remove 2π jumps

# Compute snapping metric as derivative
dtheta_dpsi = np.gradient(theta_tip_vals, np.deg2rad(psi_deg_range))

# Plot results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(psi_deg_range, np.rad2deg(theta_tip_vals))
plt.xlabel("ψ₁ (deg)")
plt.ylabel("Tip Angle θ(L) [deg]")
plt.title("Final Bending Angle vs Magnetization Angle")

plt.subplot(1, 2, 2)
plt.plot(psi_deg_range, dtheta_dpsi)
plt.xlabel("ψ₁ (deg)")
plt.ylabel("dθ/dψ₁")
plt.title("Snapping Sensitivity (Derivative)")

plt.tight_layout()
plt.show()
