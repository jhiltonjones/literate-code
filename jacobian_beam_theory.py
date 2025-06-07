import math
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.optimize import minimize
import time

L = 0.05  # rod length in meters (24 mm)
r = 0.00054  # rod radius in meters (0.54 mm)
E = 3.0e6  
A = math.pi * r**2
I = math.pi * r**4 / 4.0

# Magnetic constants for the external magnet (point dipole model)
MU0 = 4 * math.pi * 1e-7      # vacuum permeability (μ0)
M = 8000

MAGNET_M = 318   


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

    first_term = f_first_term(theta_val, x_val, y_val, x_m, y_m, magnet_dipole_angle, C_val)
    second_term = f_second_term(theta_val, x_val, y_val, x_m, y_m, magnet_dipole_angle, C_val)

    total_symbolic = first_term + second_term

    return total_symbolic
# --- Step 3: Updated deflection solver using full symbolic dF/dtheta ---
def solve_deflection_angle(magnet_pos, magnet_dipole_angle, n_steps = 1000):
    def integrate_curvature(k0):
        theta = 0.0
        dtheta = k0
        x, y = 0.0, 0.0
        n_steps = 1000
        ds = L / n_steps
        for _ in range(n_steps):
            ddtheta = -(A * M / (E * I)) * compute_dF_dtheta_symbolic(theta, x, y, magnet_pos, magnet_dipole_angle)
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
        if attempts > 50:
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
        ddtheta = -(A * M / (E * I)) * compute_dF_dtheta_symbolic(theta, x, y, magnet_pos, magnet_dipole_angle)
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

def compute_Jp_fd_based(theta_vals, magnet_2d, psi, eps=1e-4):
    # Perturb magnet x
    dx = np.array([eps, 0.0])
    _, theta_vals_x, _, _ = solve_deflection_angle(magnet_2d + dx, psi)
    dtheta_dx = (np.array(theta_vals_x) - np.array(theta_vals)) / eps

    # Perturb magnet y
    dy = np.array([0.0, eps])
    _, theta_vals_y, _, _ = solve_deflection_angle(magnet_2d + dy, psi)
    dtheta_dy = (np.array(theta_vals_y) - np.array(theta_vals)) / eps

    # Project FD gradient onto m_g
    mg_vec = np.array([-np.cos(psi), np.sin(psi)])
    Jp_fd_proj = dtheta_dx[-1] * mg_vec[0] + dtheta_dy[-1] * mg_vec[1]
    return Jp_fd_proj

def compute_Jp(theta_vals, x_vals, y_vals, s_vals, x_m, y_m, psi, A, E, I, mu0, M):
    """
    Compute J_p (scalar projection of Jacobian of tip angle wrt magnet position)
    and J_psi (Jacobian wrt magnet angle psi).
    """
    N = len(s_vals) - 1
    ds = s_vals[1] - s_vals[0]
    m_g = np.array([-np.cos(psi), np.sin(psi)])
    dm_dpsi = np.array([np.sin(psi), np.cos(psi)])

    accumulated_integral = np.zeros(2)
    contrib_proj_list = []
    J_psi = 0.0

    for i in range(N + 1):
        theta_i = theta_vals[i]
        x_i = x_vals[i]
        y_i = y_vals[i]

        px = x_i - x_m
        py = y_i - y_m
        r2 = px**2 + py**2
        r = np.sqrt(r2)
        if r < 1e-8:
            continue

        p_hat = np.array([px, py]) / r
        m_hat = np.array([np.cos(psi), np.sin(psi)])
        dot_pm = np.dot(p_hat, m_hat)

        outer_pp = np.outer(p_hat, p_hat)
        outer_pm = np.outer(p_hat, m_hat)
        outer_mp = np.outer(m_hat, p_hat)
        Z = np.eye(2) - 5 * outer_pp
        C_grad = 3 * mu0 * M / (4 * np.pi * r**4)
        grad_b = C_grad * (outer_pm + dot_pm * np.eye(2) + Z @ outer_mp)

        dR_dtheta = np.array([-np.sin(theta_i), np.cos(theta_i)])
        dx_dtheta = np.array([-np.sin(theta_i), np.cos(theta_i)])
        term_dx = grad_b @ dx_dtheta

        m_local = np.array([np.cos(theta_i), np.sin(theta_i)])
        # Compute magnetic field vector B at this point
        dot_pm = np.dot(p_hat, m_hat)
        b_vec = (MU0 * M / (4 * np.pi * r**3)) * (3 * dot_pm * p_hat - m_hat)

        # dR/dtheta ⋅ B
        term_R = np.dot(dR_dtheta, b_vec)


        integrand = term_dx + term_R
        accumulated_integral += integrand * ds

        # Outer integral contribution
        contrib_proj = np.dot(accumulated_integral, m_g)
        contrib_proj_list.append(contrib_proj)

        # Also update J_psi
        J_psi += np.dot(accumulated_integral, dm_dpsi) * ds

    # Final trapezoidal integration over s
    from scipy.integrate import trapezoid
    J_p = -trapezoid(contrib_proj_list, x=s_vals)

    print(f"Final symbolic J_p = {J_p:.6f}, J_psi = {J_psi:.6f}")
    return float(J_p), float(J_psi)
def m_g(psi):
    return np.array([-np.cos(psi), np.sin(psi)])

magnet_3d = np.array([0.1, 0.03, 0.04])  # x, y, z
catheter_base = np.array([0.05, 0.0, 0.0])

phi = compute_phi_from_yz(magnet_3d[1], magnet_3d[2])

# Step 2: Project 3D magnet into 2D solver plane
magnet_2d = project_magnet_to_2d(magnet_3d[0], magnet_3d[1], magnet_3d[2])

# Step 3: Solve in 2D
# psi = np.arctan2(magnet_2d[1] - 0.0, magnet_2d[0] - 0.0)
delta = np.deg2rad(0)  # or any desired angle in radians

psi = np.arctan2(magnet_2d[1], magnet_2d[0]) + delta
print(f"Aligned psi value is {np.rad2deg(psi)}")
# psi=3.141592653589793
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

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.plot(path_3d[:, 0], path_3d[:, 1], path_3d[:, 2], label="Deflected Shape")

# # Magnet
# ax.scatter(*magnet_3d, color='red', s=50, label="Magnet", marker='X')

# # Highlight base (first point)
# ax.scatter(*path_3d[0], color='blue', s=60, label='Base', marker='o')

# # Highlight tip (last point)
# ax.scatter(*path_3d[-1], color='green', s=60, label='Tip', marker='^')

# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.legend()
# def set_axes_equal(ax):
#     '''Make 3D plot axes have equal scale.'''
#     limits = np.array([
#         ax.get_xlim3d(),
#         ax.get_ylim3d(),
#         ax.get_zlim3d(),
#     ])
#     spans = limits[:, 1] - limits[:, 0]
#     centers = np.mean(limits, axis=1)
#     max_span = max(spans)

#     new_limits = np.array([
#         center - max_span / 2 for center in centers
#     ]), np.array([
#         center + max_span / 2 for center in centers
#     ])
    
#     ax.set_xlim3d(new_limits[0][0], new_limits[1][0])
#     ax.set_ylim3d(new_limits[0][1], new_limits[1][1])
#     ax.set_zlim3d(new_limits[0][2], new_limits[1][2])

# set_axes_equal(ax)
# plt.tight_layout()
# plt.show()
# def set_axes_equal(ax):
#     '''Make 3D plot axes have equal scale.'''
#     limits = np.array([
#         ax.get_xlim3d(),
#         ax.get_ylim3d(),
#         ax.get_zlim3d(),
#     ])
#     spans = limits[:, 1] - limits[:, 0]
#     centers = np.mean(limits, axis=1)
#     max_span = max(spans)

#     new_limits = np.array([
#         center - max_span / 2 for center in centers
#     ]), np.array([
#         center + max_span / 2 for center in centers
#     ])
    
#     ax.set_xlim3d(new_limits[0][0], new_limits[1][0])
#     ax.set_ylim3d(new_limits[0][1], new_limits[1][1])
#     ax.set_zlim3d(new_limits[0][2], new_limits[1][2])

# set_axes_equal(ax)
# plt.tight_layout()
# plt.show()
# Jp, J_psi = compute_Jp(
#     theta_vals=theta_vals,
#     x_vals=x_vals,
#     y_vals=y_vals,
#     s_vals=s_vals,
#     x_m=magnet_3d[0],
#     y_m=magnet_3d[1],
#     psi=psi,
#     A=A,
#     E=E,
#     I=I,
#     mu0=MU0,
#     M=M
# )
# eps = 1e-4

# # Check J_psi via finite difference
# psi_perturbed = psi + eps
# _, theta_vals_eps, _, _ = solve_deflection_angle(magnet_2d, psi_perturbed)
# theta_eps = theta_vals_eps[-1]
# theta_nominal = theta_vals[-1]
# J_psi_fd = (theta_eps - theta_nominal) / eps

# print(f"J_psi (FD)     = {J_psi_fd:.6f} rad/rad")
# print(f"J_psi (Symbol) = {J_psi:.6f} rad/rad")
# print(f"Error J_psi    = {abs(J_psi - J_psi_fd):.2e}")

# # Check J_p via x-direction perturbation
# magnet_dx = magnet_2d + np.array([eps, 0.0])
# s_, theta_dx, _, _ = solve_deflection_angle(magnet_dx, psi)
# Jp_fd_x = (theta_dx[-1] - theta_nominal) / eps

# # y-direction
# magnet_dy = magnet_2d + np.array([0.0, eps])
# s_, theta_dy, _, _ = solve_deflection_angle(magnet_dy, psi)
# Jp_fd_y = (theta_dy[-1] - theta_nominal) / eps

# # Project numerical Jp onto m_g
# m_g = np.array([-np.cos(psi), np.sin(psi)])
# Jp_fd_proj = Jp_fd_x * m_g[0] + Jp_fd_y * m_g[1]
# Jp_fd_proj = compute_Jp_fd_based(theta_vals, magnet_2d, psi)

# print(f"J_p (FD proj)  = {Jp_fd_proj:.6f} rad/m")
# print(f"J_p (Symbol)   = {Jp:.6f} rad/m")
# print(f"Error J_p      = {abs(Jp - Jp_fd_proj):.2e}")
# print(f"FD Jp_x = {Jp_fd_x:.6f}, FD Jp_y = {Jp_fd_y:.6f}")
# print(f"m_g = {m_g}")
# print(f"FD projection = {Jp_fd_proj:.6f}")

# Newton-based IK solver
def inverse_kinematics(theta_target_debug, max_iters=15, tol=np.deg2rad(3)):
    def run_attempt(x_init, y_init, psi_init, first_time, theta_target_sol, start_time):
        x, y, psi = x_init, y_init, psi_init
        best_error = float("inf")
        best_solution = (x, y, psi, None)
        no_improve_count = 0
        max_no_improve = 2

        for i in range(max_iters):
            if time.time() - start_time > 90:
                print(f"[INFO] Timeout reached after 1 minute. Returning best solution found.")
                break
            s_vals, theta_vals, x_vals, y_vals = solve_deflection_angle((x, y), psi)
            theta_tip = theta_vals[-1]
            error = theta_tip - theta_target_sol

            if abs(error) < abs(best_error):
                best_error = error
                best_solution = (x, y, psi, theta_tip)
                no_improve_count = 0
            else:
                no_improve_count += 1

            if abs(error) < tol:
                break

            Jp_val, Jpsi_val = compute_Jp(theta_vals, x_vals, y_vals, s_vals, x, y, psi, A, E, I, MU0, M)
            mg = m_g(psi)
            J_total = np.array([Jpsi_val, Jp_val * mg[0], Jp_val * mg[1]])
            denom = np.dot(J_total, J_total)

            if denom < 1e-8:
                print(f"[WARNING] Jacobian too small at iter {i}, skipping update.")
                break

            update = -error * J_total / (denom + 1e-3)
            alpha = 1.0
            accepted = False
            error_trial = error  # fallback default in case all trials fail

            for j in range(10):
                trial_alpha = alpha * (0.5 ** j)

                x_trial = np.clip(x + trial_alpha * update[1], -0.25, 0.25)
                y_trial = np.clip(y + trial_alpha * update[2], -0.15, 0.15)
                psi_trial = (psi + trial_alpha * update[0] + np.pi) % (2 * np.pi) - np.pi

                try:
                    _, theta_vals_trial, _, _ = solve_deflection_angle((x_trial, y_trial), psi_trial)
                    error_trial = theta_vals_trial[-1] - theta_target_sol
                except Exception:
                    continue

                if abs(error_trial) < abs(error):
                    x, y, psi = x_trial, y_trial, psi_trial
                    print(f"[iter {i}] error = {np.rad2deg(error):.2f}° → {np.rad2deg(error_trial):.2f}°, accepted with α={trial_alpha:.4f}")
                    accepted = True
                    break

            # Handle fallback mirror strategy BEFORE forcing a small step
            if abs(np.rad2deg(error_trial)) > 40 and first_time == 0:
                print(f"[INFO] Error {np.rad2deg(error):.2f}° too large at iter {i}. Switching initial guess...")
                new_psi = np.arctan2(0.01, 0.12)
                return run_attempt(0.01, 0.12, new_psi, 1, theta_target_sol, start_time)
            if abs(np.rad2deg(error_trial)) > 40 and first_time == 1:
                print(f"[INFO] Error {np.rad2deg(error):.2f}° too large at iter {i}. Switching initial guess again...")
                new_psi = np.arctan2(0.06, 0.08)
                return run_attempt(0.06, 0.08, new_psi, 2, theta_target_sol, start_time)

            if not accepted:
                print(f"[iter {i}] No improvement after decay, forcing small update.")
                trial_alpha = 1e-2
                x += trial_alpha * update[1]
                y += trial_alpha * update[2]
                psi += trial_alpha * update[0]
                x = np.clip(x, 0.01, 0.25)
                y = np.clip(y, 0.01, 0.15)
                psi = psi % (2 * np.pi)

        return best_solution


    # Detect if flipping strategy should be used
    flip = theta_target_debug < np.deg2rad(-10)
    first_time = 0
    theta_target_sol = abs(theta_target_debug)
    psi_direct = np.arctan2(0.1, 0.02)
    x_init, y_init, psi_init = 0.1, 0.02, psi_direct
    start_time = time.time()
    x_best, y_best, psi_best, theta_best = run_attempt(x_init, y_init, psi_init, first_time, theta_target_sol, start_time)

    if flip:
        y_best = -1 * y_best
        psi_best = (-psi_best + np.pi) % (2 * np.pi) - np.pi

    return x_best, y_best, psi_best, np.rad2deg(theta_best), np.rad2deg(theta_target_debug)




# theta_target2 = (np.deg2rad(15))
# x_solved, y_solved, psi_solved, tehta, target = inverse_kinematics(theta_target2)
# print(x_solved, y_solved, np.rad2deg(psi_solved), tehta, target)



def solve_for_target_3d(target_3d):
    """
    target_3d: array-like [x, y, z], desired 3D tip location

    Returns:
        dict with all data for 3D and 2D plotting and tip analysis
    """
    x3, y3, z3 = target_3d
    r_yz = np.sqrt(y3**2 + z3**2)
    theta_target = np.arctan2(r_yz, x3)
    phi = np.arctan2(z3, y3)

    print(f"[INFO] Target tip: x={x3:.3f}, y={y3:.3f}, z={z3:.3f}")
    print(f"[INFO] Target bending angle θ = {np.rad2deg(theta_target):.2f}°, direction φ = {np.rad2deg(phi):.2f}°")

    # Solve inverse kinematics in 2D
    x_solved, y_solved, psi_solved, theta_sol, theta_target_used = inverse_kinematics(theta_target)

    # Forward model in 2D
    s_vals, theta_vals, x_vals, y_vals = solve_deflection_angle((x_solved, y_solved), psi_solved)

    # Convert to 3D
    path_3d = rotate_2d_to_3d(x_vals, y_vals, phi)
    magnet_pos_3d = rotate_2d_to_3d(np.array([x_solved]), np.array([y_solved]), phi)[0]
    actual_tip = path_3d[-1]

    return {
        "path_3d": path_3d,
        "magnet_pos_3d": magnet_pos_3d,
        "x_vals": x_vals,
        "y_vals": y_vals,
        "psi_solved": psi_solved,
        "x_solved": x_solved,
        "y_solved": y_solved,
        "target_3d": target_3d,
        "actual_tip": actual_tip,
        "phi": phi,
        "theta_target_deg": np.rad2deg(theta_target),
        "theta_solved_deg": np.rad2deg(theta_sol)
    }



target = np.array([final_position_3d[0], final_position_3d[1], final_position_3d[2]])  
res = solve_for_target_3d(target)


# psi_vals = np.linspace(-np.pi, np.pi, 10)
# J_psi_vals = []
# Jp_vals = []

# for psi_i in psi_vals:
#     s_, theta_i, x_, y_ = solve_deflection_angle(magnet_2d, psi_i)
#     Jp_i, Jpsi_i = compute_Jp(
#         np.array([1.0, 0.0]), theta_i, x_, y_, s_, 
#         magnet_2d[0], magnet_2d[1], psi_i, A, E, I, MU0, M)
#     J_psi_vals.append(Jpsi_i)
#     Jp_vals.append(Jp_i)

# plt.figure(figsize=(10, 4))
# plt.plot(psi_vals, J_psi_vals, label="J_psi")
# plt.plot(psi_vals, Jp_vals, label="J_p")
# plt.xlabel("ψ (rad)")
# plt.ylabel("Jacobian value")
# plt.legend()
# plt.title("Jacobian trends with respect to ψ")
# plt.grid(True)
# plt.show()

# phi = np.deg2rad(45)  # 45 degrees elevation in the YZ-plane


# inversekinematics_sol = np.array([x_solved, y_solved])
# # inversekinematics_sol_mag = np.array([x_solved, y_solved, phi])
# s_vals, theta_vals, x_vals, y_vals = solve_deflection_angle(inversekinematics_sol, psi_solved)

# # Step 4: Rotate the 2D result back into 3D
# path_3d = rotate_2d_to_3d(x_vals, y_vals, phi)
# # Print final bending angle (in degrees)
# final_bending_rad = theta_vals[-1]
# final_bending_deg = np.rad2deg(final_bending_rad)
# print(f"Final bending angle θ(L): {final_bending_deg:.3f} degrees")

# # Print final position in 3D (x, y, z)
# final_position_3d = path_3d[-1]
# print(f"Final tip position: x = {final_position_3d[0]:.4f}, y = {final_position_3d[1]:.4f}, z = {final_position_3d[2]:.4f}")
# magnet_pos_2d = np.array([x_solved, y_solved])
# magnet_pos_3d = rotate_2d_to_3d(np.array([magnet_pos_2d[0]]), np.array([magnet_pos_2d[1]]), phi)[0]
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(path_3d[:, 0], path_3d[:, 1], path_3d[:, 2], label="Deflected Shape")

# Plot the magnet
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
# plt.figure(figsize=(6, 6))

# # Plot catheter deformation
# plt.plot(x_vals, y_vals, linewidth=2, label=f'ψ = {math.degrees(psi_solved):.1f}°')

# # Plot magnet position
# plt.scatter(x_solved, y_solved, color='red', s=80, label='Magnet')
# plt.text(x_solved, y_solved, '  Magnet', color='red', fontsize=10, va='bottom')

# # Axes and styling
# plt.xlabel('x (m)')
# plt.ylabel('y (m)')
# plt.title('MSCR Deformation Under Magnetic Actuation')
# plt.axis('equal')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()
