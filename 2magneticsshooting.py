import math
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
magnet_position = np.array([-0.02, 0.15])
import numpy as np

def compute_phi_from_yz(y_mag, z_mag):
    # Avoid division by zero
    if y_mag == 0 and z_mag == 0:
        return 0.0
    return np.arctan2(z_mag, y_mag)  # angle between y-axis and magnet in the YZ plane

def project_magnet_to_2d(x_mag, y_mag, z_mag):
    y_proj = np.sqrt(y_mag**2 + z_mag**2)  # radial projection onto the YZ plane
    return np.array([x_mag, y_proj])

def rotate_2d_to_3d(x_vals, y_vals, phi):
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    
    # Rotate around the X-axis by angle φ
    y_3d = y_vals * np.cos(phi)
    z_3d = y_vals * np.sin(phi)
    
    path_3d = np.stack((x_vals, y_3d, z_3d), axis=1)
    return path_3d
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

def solve_deflection_angle_newton_safe(magnet_pos, magnet_dipole_angle, n_steps=1000):
    ds = L / n_steps
    max_newton_iters = 200

    # --- Bracketing method to find a good initial guess ---
    def integrate_curvature(k0):

        theta = 0.0
        dtheta = k0
        x = y = 0.0
        for _ in range(n_steps):
            if _ < n_steps // 2:
                local_dipole_angle = magnet_dipole_angle
            else:
                local_dipole_angle = magnet_dipole_angle   # flipped magnetization of the rod

            ddtheta = -(A * M / (E * I)) * compute_dF_dtheta_symbolic(theta, x, y, magnet_pos, local_dipole_angle)
            dx = np.cos(theta)
            dy = np.sin(theta)
            theta += dtheta * ds
            dtheta += ddtheta * ds
            x += dx * ds
            y += dy * ds

        return dtheta  # return θ'(L)

    # Bracket the root
    k0_low, k0_high = 0.0, 50.0
    res_low = integrate_curvature(k0_low)
    res_high = integrate_curvature(k0_high)

    attempts = 0
    while res_low * res_high > 0:

        k0_low = k0_high
        res_low = res_high
        k0_high *= 2
        res_high = integrate_curvature(k0_high)
        attempts += 1
        if attempts > 100:
            print("[WARNING] No sign change in integrate_curvature. Switching to fallback k0 range.")
            k0_low, k0_high = -50, 50
            res_low = integrate_curvature(k0_low)
            res_high = integrate_curvature(k0_high)
            break

    # Use midpoint as Newton initial guess
    v = 0.5 * (k0_low + k0_high)

    # --- Newton iteration ---
    def integrate_state_and_sensitivity(v):
        theta = 0.0
        dtheta = v
        z = 0.0
        dz = 1.0
        x = y = 0.0
        for _ in range(n_steps):
            if _ < n_steps // 2:
                local_dipole_angle = magnet_dipole_angle
            else:
                local_dipole_angle = magnet_dipole_angle + np.pi  # flipped magnetization of the rod

            dF_dtheta = compute_dF_dtheta_symbolic(theta, x, y, magnet_pos, local_dipole_angle)

            ddtheta = -(A * M / (E * I)) * dF_dtheta
            # Finite difference to approximate derivative of ddtheta
            dF_dtheta_eps = compute_dF_dtheta_symbolic(theta + 1e-6, x, y, magnet_pos, local_dipole_angle)

            dddtheta_dtheta = -(A * M / (E * I)) * (dF_dtheta_eps - dF_dtheta) / 1e-6
            # Integrate θ and sensitivity z = ∂θ/∂v
            theta += dtheta * ds
            dtheta += ddtheta * ds
            z += dz * ds
            dz += dddtheta_dtheta * z * ds
            # Integrate position
            dx = math.cos(theta)
            dy = math.sin(theta)
            x += dx * ds
            y += dy * ds
        return theta, dtheta, z, dz

    success = False
    for _ in range(max_newton_iters):

        theta_L, dtheta_L, z_L, dz_L = integrate_state_and_sensitivity(v)
        residual = dtheta_L
        if abs(residual) < 1e-6:
            success = True
            break
        if abs(dz_L) < 1e-6:
            print("[WARNING] Newton step sensitivity too small. Aborting Newton updates.")
            break
        v -= residual / dz_L
        # print(f"[DEBUG] Iter {_}: v = {v:.6f}, dtheta_L = {dtheta_L:.6e}, dz_L = {dz_L:.6e}")

    if not success:
        print("[WARNING] Newton did not converge. Falling back to v = 1.0")
        v = 1.0

    # --- Final integration to extract the curve ---
    theta = 0.0
    dtheta = v
    x = y = 0.0
    theta_vals = [theta]
    x_vals = [x]
    y_vals = [y]
    for _ in range(n_steps):
        ddtheta = -(A * M / (E * I)) * compute_dF_dtheta_symbolic(theta, x, y, magnet_pos, magnet_dipole_angle)
        theta += dtheta * ds
        dtheta += ddtheta * ds
        dx = math.cos(theta)
        dy = math.sin(theta)
        x += dx * ds
        y += dy * ds
        theta_vals.append(theta)
        x_vals.append(x)
        y_vals.append(y)

    s_vals = [i * ds for i in range(len(theta_vals))]
    return s_vals, theta_vals, x_vals, y_vals


# --- Step 3: Updated deflection solver using full symbolic dF/dtheta ---
def solve_deflection_angle(magnet_pos, magnet_dipole_angle, n_steps = 1000):
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
                local_dipole_angle = magnet_dipole_angle  # -x direction
            
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
            local_dipole_angle = magnet_dipole_angle  # +x direction
        elif _ < 3 * n_steps // 4:
            local_dipole_angle = magnet_dipole_angle    # +x direction
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

# Original 3D magnet position
magnet_3d = np.array([0.08, 0.03, 0.0])  # x, y, z
catheter_base = np.array([0.05, 0.0, 0.0])

phi = compute_phi_from_yz(magnet_3d[1], magnet_3d[2])

# Step 2: Project 3D magnet into 2D solver plane
magnet_2d = project_magnet_to_2d(magnet_3d[0], magnet_3d[1], magnet_3d[2])

# Step 3: Solve in 2D
# psi = np.arctan2(magnet_2d[1] - 0.0, magnet_2d[0] - 0.0)
delta = np.deg2rad(0)  # or any desired angle in radians

psi = np.arctan2(magnet_2d[1], magnet_2d[0]) + delta

s_vals, theta_vals, x_vals, y_vals = solve_deflection_angle(magnet_2d, psi)

# Step 4: Rotate the 2D result back into 3D
path_3d = rotate_2d_to_3d(x_vals, y_vals, phi)
# Print final bending angle (in degrees)
final_bending_rad = theta_vals[-1]
final_bending_deg = np.rad2deg(final_bending_rad)
print(f"Final bending angle θ(L): {final_bending_deg:.3f} degrees")

# Print final position in 3D (x, y, z)
final_position_3d = path_3d[-1]
print(f"Final tip position: x = {final_position_3d[0]:.4f}, y = {final_position_3d[1]:.4f}, z = {final_position_3d[2]:.4f}")

# Plot the result
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np  # Ensure this is imported if not already

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(path_3d[:, 0], path_3d[:, 1], path_3d[:, 2], label="Deflected Shape")

# Magnet
ax.scatter(*magnet_3d, color='red', s=50, label="Magnet", marker='X')

# Highlight base (first point)
ax.scatter(*path_3d[0], color='blue', s=60, label='Base', marker='o')

# Highlight tip (last point)
ax.scatter(*path_3d[-1], color='green', s=60, label='Tip', marker='^')

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()

def set_axes_equal(ax):
    '''Make 3D plot axes have equal scale.'''
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    spans = limits[:, 1] - limits[:, 0]
    centers = np.mean(limits, axis=1)
    max_span = max(spans)

    new_limits = np.array([
        center - max_span / 2 for center in centers
    ]), np.array([
        center + max_span / 2 for center in centers
    ])
    
    ax.set_xlim3d(new_limits[0][0], new_limits[1][0])
    ax.set_ylim3d(new_limits[0][1], new_limits[1][1])
    ax.set_zlim3d(new_limits[0][2], new_limits[1][2])

set_axes_equal(ax)
plt.tight_layout()
plt.show()

# Example usage (one input scenario):
# Magnet located 0.25 m above the base (ρ_G = 0.25 m, angle 90°), with dipole rotated by ψ = -90° (pointing downward).
# magnet_position = (0.02, 0.2)
# print(f"Magnet position: {magnet_position}")
# catheter_pos = np.array([0.08, 0])  # Catheter starts at origin

# # Direction vector from catheter to magnet
# direction = magnet_position - catheter_pos

# # Compute angle in radians and degrees
# angle_rad = np.arctan2(direction[1], direction[0])
# angle_deg = np.rad2deg(angle_rad)

# # reverse_angle = -1* angle_deg

# # angle_custom = np.deg2rad(90)
# # psi = np.deg2rad(reverse_angle)
# # psi = angle_custom
# psi = angle_rad
# print(f"angel of mag {angle_deg}")
# s_vals, theta_vals, x_vals, y_vals = solve_deflection_angle(magnet_position, psi)

# import matplotlib.pyplot as plt

# plt.figure(figsize=(6, 6))

# # Plot catheter deformation
# plt.plot(x_vals, y_vals, linewidth=2, label=f'ψ = {math.degrees(psi):.1f}°')

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

# import numpy as np
# import matplotlib.pyplot as plt
# import math

# plt.figure(figsize=(8, 6))

# # Loop through psi from -90° to 90° in steps of 20°
# for deg in range(0, 360, 20):
#     psi = math.radians(deg)
#     print(f"\nψ = {deg}°")

#     # Compute deflection and Jacobian
#     s_vals, theta_vals, x_vals, y_vals = solve_deflection_angle(magnet_position, psi)

#     tip_angle = theta_vals[-1]

#     print(f"  Tip deflection angle θ(L) = {np.rad2deg(tip_angle):.2f}°")

#     # Plot the catheter deformation for this ψ
#     plt.plot(x_vals, y_vals, label=f'ψ = {deg}°')
#     # Add small label at the tip of the curve
#     tip_x, tip_y = x_vals[-1], y_vals[-1]
#     plt.text(tip_x + 0.001, tip_y + 0.001, f'{deg}°', fontsize=8, alpha=0.8)

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

# # Final plot styling
# plt.xlabel('x (m)')
# plt.ylabel('y (m)')
# plt.title('MSCR Deformation for Various Magnet Orientations at Magnet position')
# plt.axis('equal')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()