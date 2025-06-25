import math
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.optimize import minimize
import time

L = 0.05  
r = 0.00054  
E = 3.0e6  
A = math.pi * r**2
I = math.pi * r**4 / 4.0

MU0 = 4 * math.pi * 1e-7      
M = 8000

MAGNET_M = 318   

def compute_phi_from_yz(y_mag, z_mag):
   
    if y_mag == 0 and z_mag == 0:
        return 0.0
    return np.arctan2(z_mag, y_mag)  

def project_magnet_to_2d(x_mag, y_mag, z_mag):
    y_proj = np.sqrt(y_mag**2 + z_mag**2) 
    return np.array([x_mag, y_proj])

def rotate_2d_to_3d(x_vals, y_vals, phi):
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    

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
        n_steps = 100
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

    k0_low, k0_high = 0.0, 50.0
    res_low = integrate_curvature(k0_low)  
    res_high = integrate_curvature(k0_high)
    
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
    k0_mid = (k0_a + k0_b) / 2  
    k0_solution = None
    for _ in range(50): 
        if abs(res_b - res_a) < 1e-9:
            break
        # Secant update for k0
        k0_mid = k0_b - res_b * ((k0_b - k0_a) / (res_b - res_a))
        res_mid = integrate_curvature(k0_mid)
        if abs(res_mid) < 1e-6:
            k0_solution = k0_mid
            break
       
        if res_a * res_mid < 0:
            k0_b, res_b = k0_mid, res_mid
        else:
            k0_a, res_a = k0_mid, res_mid
        k0_solution = k0_mid
    if k0_solution is None:
        k0_solution = k0_mid  

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

def compute_full_gradient(theta_vals, x_vals, y_vals, s_vals, x_m, y_m, psi, A, E, I, mu0, M):
    """
    Compute the full gradient of tip angle wrt magnet position (∇_{p_m} θ_tip)
    and the derivative wrt magnet angle psi (J_psi).
    """
    N = len(s_vals) - 1
    ds = s_vals[1] - s_vals[0]
    dm_dpsi = np.array([np.sin(psi), np.cos(psi)])

    accumulated_integral = np.zeros(2)
    integrand_list = []
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

        b_vec = (mu0 * M / (4 * np.pi * r**3)) * (3 * dot_pm * p_hat - m_hat)
        term_R = np.dot(dR_dtheta, b_vec)

        integrand = term_dx + term_R
        accumulated_integral += integrand * ds
        integrand_list.append(integrand.copy())

        J_psi += np.dot(accumulated_integral, dm_dpsi) * ds

    from scipy.integrate import trapezoid
    grad_x = trapezoid([vec[0] for vec in integrand_list], x=s_vals)
    grad_y = trapezoid([vec[1] for vec in integrand_list], x=s_vals)

    grad_vector = -np.array([grad_x, grad_y]) 

    print(f"Full gradient ∇θ_tip = {grad_vector}, J_psi = {J_psi:.6f}")
    return grad_vector, float(J_psi)

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
        dot_pm = np.dot(p_hat, m_hat)
        b_vec = (MU0 * M / (4 * np.pi * r**3)) * (3 * dot_pm * p_hat - m_hat)

        # dR/dtheta ⋅ B
        term_R = np.dot(dR_dtheta, b_vec)


        integrand = term_dx + term_R
        accumulated_integral += integrand * ds

        contrib_proj = np.dot(accumulated_integral, m_g)
        contrib_proj_list.append(contrib_proj)

        J_psi += np.dot(accumulated_integral, dm_dpsi) * ds

    from scipy.integrate import trapezoid
    J_p = -trapezoid(contrib_proj_list, x=s_vals)

    print(f"Final symbolic J_p = {J_p:.6f}, J_psi = {J_psi:.6f}")
    return float(J_p), float(J_psi)
def m_g(psi):
    return np.array([-np.cos(psi), np.sin(psi)])

# magnet_3d = np.array([0.09, -0.04, 0.02]) 
# catheter_base = np.array([0.05, 0.0, 0.0])
magnet_3d = np.array([0.1, 0.03, 0.04]) 
catheter_base = np.array([0.05, 0.0, 0.0])
phi = compute_phi_from_yz(magnet_3d[1], magnet_3d[2])

magnet_2d = project_magnet_to_2d(magnet_3d[0], magnet_3d[1], magnet_3d[2])

# psi = np.arctan2(magnet_2d[1] - 0.0, magnet_2d[0] - 0.0)
delta = np.deg2rad(0)  
# psi = -1.36
psi = np.arctan2(magnet_2d[1], magnet_2d[0]) + delta
print(f"Aligned psi value is {np.rad2deg(psi)}")
# psi=3.141592653589793
s_vals, theta_vals, x_vals, y_vals = solve_deflection_angle(magnet_2d, psi)

path_3d = rotate_2d_to_3d(x_vals, y_vals, phi)
final_bending_rad = theta_vals[-1]
final_bending_deg = np.rad2deg(final_bending_rad)
print(f"Final bending angle θ(L): {final_bending_deg:.3f} degrees")

final_position_3d = path_3d[-1]
print(f"Final tip position: x = {final_position_3d[0]:.4f}, y = {final_position_3d[1]:.4f}, z = {final_position_3d[2]:.4f}")

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


import numpy as np
import matplotlib.pyplot as plt

# Your helper functions must be defined:
# compute_phi_from_yz, project_magnet_to_2d, solve_deflection_angle, rotate_2d_to_3d

# List of 3D magnet positions
magnet_list = [
    np.array([0.05, -0.08, 0.07]),
    np.array([-0.03, 0.03, 0.1]),
    np.array([0.1, 0.03, 0.04])
]

catheter_base = np.array([0.05, 0.0, 0.0])
delta = np.deg2rad(0)

# Prepare plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Use Matplotlib default color cycle
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i, magnet_3d in enumerate(magnet_list):
    color = colors[i % len(colors)]

    phi = compute_phi_from_yz(magnet_3d[1], magnet_3d[2])
    magnet_2d = project_magnet_to_2d(magnet_3d[0], magnet_3d[1], magnet_3d[2])
    psi = np.arctan2(magnet_2d[1], magnet_2d[0]) + delta

    s_vals, theta_vals, x_vals, y_vals = solve_deflection_angle(magnet_2d, psi)
    path_3d = rotate_2d_to_3d(x_vals, y_vals, phi)

    # Plot deflected path in unique color
    ax.plot(path_3d[:, 0], path_3d[:, 1], path_3d[:, 2],
            label=f"Deflected Shape {i+1}", color=color, lw=2)

    # Magnet in matching color
    ax.scatter(*magnet_3d, color=color, s=50, marker='X', label=f"Magnet {i+1}")

    # Base and tip (optional: use same color but smaller markers)
    ax.scatter(*path_3d[0], color=color, s=40, marker='o')  # base
    ax.scatter(*path_3d[-1], color=color, s=40, marker='^')  # tip

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()

def set_axes_equal(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    spans = limits[:, 1] - limits[:, 0]
    centers = np.mean(limits, axis=1)
    max_span = max(spans)
    ax.set_xlim3d(centers[0] - max_span / 2, centers[0] + max_span / 2)
    ax.set_ylim3d(centers[1] - max_span / 2, centers[1] + max_span / 2)
    ax.set_zlim3d(centers[2] - max_span / 2, centers[2] + max_span / 2)

set_axes_equal(ax)
plt.tight_layout()
plt.show()






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

def inverse_kinematics(theta_target_debug, target_tip=None, max_iters=15, tol=np.deg2rad(3), use_position=False):
    def run_attempt(x_init, y_init, psi_init, first_time, theta_target_sol, start_time, target_tip_3d=None):
        x, y, psi = x_init, y_init, psi_init
        best_error = float("inf")
        best_solution = (x, y, psi, None)

        for i in range(max_iters):
            if time.time() - start_time > 200:
                print(f"[INFO] Timeout reached after 1.5 min. Returning best solution.")
                break

            s_vals, theta_vals, x_vals, y_vals = solve_deflection_angle((x, y), psi)
            theta_tip = theta_vals[-1]

            if use_position:
                path_3d = rotate_2d_to_3d(x_vals, y_vals, phi)
                tip = path_3d[-1]
                error = np.linalg.norm(tip - target_tip_3d)
                print(f"Error is : {error}")
            else:
                error = theta_tip - theta_target_sol

            if abs(error) < abs(best_error):
                best_error = error
                best_solution = (x, y, psi, theta_tip)
            else:
                pass
            if not use_position and abs(error) > 40:
                print(f"Switching intial position")
                psi_direct2 = np.arctan2(0.05, 0.1)
                run_attempt(
                            0.05, 0.1, psi_direct2, first_time, theta_target_sol, start_time
                        )
            if not use_position and abs(error) < tol:
                break
            if use_position and abs(error) < 0.005:
                break
            Jp_val, Jpsi_val = compute_Jp(theta_vals, x_vals, y_vals, s_vals, x, y, psi, A, E, I, MU0, M)
            mg = m_g(psi)
            J_total = np.array([Jpsi_val, Jp_val * mg[0], Jp_val * mg[1]])
            denom = np.dot(J_total, J_total)

            if denom < 1e-8:
                print(f"[WARNING] Jacobian too small at iter {i}, skipping update.")
                break

            if not use_position and abs(np.rad2deg(error)) > 10:
                update = np.array([0.0, -error * J_total[1], -error * J_total[2]])
            else:
                update = -error * J_total / (denom + 1e-3)

            if use_position and abs(np.rad2deg(error)) <0.0001 :
                update = np.array([0.0, -error * J_total[1], -error * J_total[2]])
            else:
                update = -error * J_total / (denom + 1e-3)
            alpha = 1.0
            accepted = False
            error_trial = error

            for j in range(5):
                trial_alpha = alpha * (0.5 ** j)
                x_trial = np.clip(x + trial_alpha * update[1], -0.25, 0.25)
                y_trial = np.clip(y + trial_alpha * update[2], -0.15, 0.15)
                psi_trial = (psi + trial_alpha * update[0] + np.pi) % (2 * np.pi) - np.pi

                try:
                    _, theta_vals_trial, x_vals_trial, y_vals_trial = solve_deflection_angle((x_trial, y_trial), psi_trial)
                    if use_position:
                        path_3d_trial = rotate_2d_to_3d(x_vals_trial, y_vals_trial, phi)
                        error_trial = np.linalg.norm(path_3d_trial[-1] - target_tip_3d)
                    else:
                        error_trial = theta_vals_trial[-1] - theta_target_sol
                except Exception:
                    continue

                if abs(error_trial) < abs(error):
                    x, y, psi = x_trial, y_trial, psi_trial
                    print(f"[iter {i}] error = {np.rad2deg(error):.2f}° → {np.rad2deg(error_trial):.2f}°, α={trial_alpha:.4f}")
                    accepted = True
                    break

            if not accepted:
                print(f"[iter {i}] Forcing small fallback step.")
                trial_alpha = 1e-2
                # trial_alpha = 10

                x += trial_alpha * update[1]
                y += trial_alpha * update[2]
                psi += trial_alpha * update[0]
                if not use_position and abs(np.rad2deg(error)) > 10:
                    psi = np.arctan2(x,y)

                x = np.clip(x, 0.01, 0.25)
                y = np.clip(y, 0.01, 0.15)
                psi = psi % (2 * np.pi)

        return best_solution
    flip = theta_target_debug < np.deg2rad(-10)
    theta_target_sol = abs(theta_target_debug)
    first_time = 0
    psi_direct = np.arctan2(0.1, 0.02)
    x_init, y_init, psi_init = 0.1, 0.02, psi_direct
    start_time = time.time()

    if use_position:
        assert target_tip is not None, "Position mode requires target_tip"
        phi = np.arctan2(target_tip[2], target_tip[1])
        x_best, y_best, psi_best, theta_best = run_attempt(
            x_init, y_init, psi_init, first_time, theta_target_sol, start_time, target_tip_3d=target_tip
        )
    else:
        x_best, y_best, psi_best, theta_best = run_attempt(
            x_init, y_init, psi_init, first_time, theta_target_sol, start_time
        )

    if flip:
        y_best = -y_best
        psi_best = (-psi_best + np.pi) % (2 * np.pi) - np.pi

    return x_best, y_best, psi_best, np.rad2deg(theta_best), np.rad2deg(theta_target_debug)
def inverse_kinematics3(theta_target_debug, target_tip=None, max_iters=15, tol=np.deg2rad(3), use_position=False):
    def run_attempt(x_init, y_init, psi_init, first_time, theta_target_sol, start_time, target_tip_3d=None):
        x, y, psi = x_init, y_init, psi_init
        best_error = float("inf")
        best_solution = (x, y, psi, None)

        for i in range(max_iters):
            if time.time() - start_time > 200:
                print(f"[INFO] Timeout reached after 1.5 min. Returning best solution.")
                break

            s_vals, theta_vals, x_vals, y_vals = solve_deflection_angle((x, y), psi)
            theta_tip = theta_vals[-1]

            if use_position:
                path_3d = rotate_2d_to_3d(x_vals, y_vals, phi)
                tip = path_3d[-1]
                error = np.linalg.norm(tip - target_tip_3d)
                print(f"Error is : {error}")
            else:
                error = theta_tip - theta_target_sol

            if abs(error) < abs(best_error):
                best_error = error
                best_solution = (x, y, psi, theta_tip)
            else:
                pass
            if not use_position and abs(error) > 40:
                print(f"Switching intial position")
                psi_direct2 = np.arctan2(0.05, 0.1)
                run_attempt(
                            0.05, 0.1, psi_direct2, first_time, theta_target_sol, start_time
                        )
            if not use_position and abs(error) < tol:
                break
            if use_position and abs(error) < 0.005:
                break
            grad_vector, Jpsi_val = compute_full_gradient(theta_vals, x_vals, y_vals, s_vals, x, y, psi, A, E, I, MU0, M)
            mg = m_g(psi)
            J_total = np.array([Jpsi_val, grad_vector[0], grad_vector[1]])  # shape (3,)
            denom = np.dot(J_total, J_total)

            if denom < 1e-8:
                print(f"[WARNING] Jacobian too small at iter {i}, skipping update.")
                break

            if not use_position and abs(np.rad2deg(error)) > 10:
                update = np.array([0.0, -error * J_total[1], -error * J_total[2]])
            else:
                update = -error * J_total / (denom + 1e-3)

            if use_position and abs(np.rad2deg(error)) <0.0001 :
                update = np.array([0.0, -error * J_total[1], -error * J_total[2]])
            else:
                update = -error * J_total / (denom + 1e-3)
            alpha = 1.0
            accepted = False
            error_trial = error

            for j in range(5):
                trial_alpha = alpha * (0.5 ** j)
                x_trial = np.clip(x + trial_alpha * update[1], -0.25, 0.25)
                y_trial = np.clip(y + trial_alpha * update[2], -0.15, 0.15)
                psi_trial = (psi + trial_alpha * update[0] + np.pi) % (2 * np.pi) - np.pi

                try:
                    _, theta_vals_trial, x_vals_trial, y_vals_trial = solve_deflection_angle((x_trial, y_trial), psi_trial)
                    if use_position:
                        path_3d_trial = rotate_2d_to_3d(x_vals_trial, y_vals_trial, phi)
                        error_trial = np.linalg.norm(path_3d_trial[-1] - target_tip_3d)
                    else:
                        error_trial = theta_vals_trial[-1] - theta_target_sol
                except Exception:
                    continue

                if abs(error_trial) < abs(error):
                    x, y, psi = x_trial, y_trial, psi_trial
                    print(f"[iter {i}] error = {np.rad2deg(error):.2f}° → {np.rad2deg(error_trial):.2f}°, α={trial_alpha:.4f}")
                    accepted = True
                    break

            if not accepted:
                print(f"[iter {i}] Forcing small fallback step.")
                trial_alpha = 1e-2
                # trial_alpha = 10

                x += trial_alpha * update[1]
                y += trial_alpha * update[2]
                psi += trial_alpha * update[0]
                if not use_position and abs(np.rad2deg(error)) > 10:
                    psi = np.arctan2(x,y)

                x = np.clip(x, 0.01, 0.25)
                y = np.clip(y, 0.01, 0.15)
                psi = psi % (2 * np.pi)

        return best_solution
    flip = theta_target_debug < np.deg2rad(-10)
    theta_target_sol = abs(theta_target_debug)
    first_time = 0
    psi_direct = np.arctan2(0.1, 0.02)
    x_init, y_init, psi_init = 0.1, 0.02, psi_direct
    start_time = time.time()

    if use_position:
        assert target_tip is not None, "Position mode requires target_tip"
        phi = np.arctan2(target_tip[2], target_tip[1])
        x_best, y_best, psi_best, theta_best = run_attempt(
            x_init, y_init, psi_init, first_time, theta_target_sol, start_time, target_tip_3d=target_tip
        )
    else:
        x_best, y_best, psi_best, theta_best = run_attempt(
            x_init, y_init, psi_init, first_time, theta_target_sol, start_time
        )

    if flip:
        y_best = -y_best
        psi_best = (-psi_best + np.pi) % (2 * np.pi) - np.pi

    return x_best, y_best, psi_best, np.rad2deg(theta_best), np.rad2deg(theta_target_debug)
def inverse_kinematics2(theta_target_debug, target_tip=None, max_iters=15, tol=np.deg2rad(3), use_position=False):
    def run_attempt(x_init, y_init, psi_init, first_time, theta_target_sol, start_time, target_tip_3d=None):
        x, y, psi = x_init, y_init, psi_init
        best_error = float("inf")
        best_solution = (x, y, psi, None)

        for i in range(max_iters):
            if time.time() - start_time > 200:
                print(f"[INFO] Timeout reached after 1.5 min. Returning best solution.")
                break

            s_vals, theta_vals, x_vals, y_vals = solve_deflection_angle((x, y), psi)
            theta_tip = theta_vals[-1]

            if use_position:
                path_3d = rotate_2d_to_3d(x_vals, y_vals, phi)
                tip = path_3d[-1]
                error = np.linalg.norm(tip - target_tip_3d)
                print(f"Error is : {error}")
            else:
                error = theta_tip - theta_target_sol

            if abs(error) < abs(best_error):
                best_error = error
                best_solution = (x, y, psi, theta_tip)
            else:
                pass
            if not use_position and abs(error) > 40:
                print(f"Switching intial position")
                psi_direct2 = np.arctan2(0.05, 0.1)
                run_attempt(
                            0.05, 0.1, psi_direct2, first_time, theta_target_sol, start_time
                        )
            if not use_position and abs(error) < tol:
                break
            if use_position and abs(error) < 0.005:
                break
            grad_vector, Jpsi_val = compute_full_gradient(theta_vals, x_vals, y_vals, s_vals, x, y, psi, A, E, I, MU0, M)

            J_total = np.array([Jpsi_val, grad_vector[0], grad_vector[1]])  # shape (3,)

            denom = np.dot(J_total, J_total)

            if denom < 1e-8:
                print(f"[WARNING] Jacobian too small at iter {i}, skipping update.")
                break

            if not use_position and abs(np.rad2deg(error)) > 10:
                update = np.array([0.0, -error * J_total[1], -error * J_total[2]])
            else:
                update = -error * J_total / (denom + 1e-3)

            if use_position and abs(np.rad2deg(error)) <0.0001 :
                update = np.array([0.0, -error * J_total[1], -error * J_total[2]])
            else:
                update = -error * J_total / (denom + 1e-3)
            alpha = 1.0
            accepted = False
            error_trial = error

            for j in range(5):
                trial_alpha = alpha * (0.5 ** j)
                x_trial = np.clip(x + trial_alpha * update[1], -0.25, 0.25)
                y_trial = np.clip(y + trial_alpha * update[2], -0.15, 0.15)
                psi_trial = (psi + trial_alpha * update[0] + np.pi) % (2 * np.pi) - np.pi

                try:
                    _, theta_vals_trial, x_vals_trial, y_vals_trial = solve_deflection_angle((x_trial, y_trial), psi_trial)
                    if use_position:
                        path_3d_trial = rotate_2d_to_3d(x_vals_trial, y_vals_trial, phi)
                        error_trial = np.linalg.norm(path_3d_trial[-1] - target_tip_3d)
                    else:
                        error_trial = theta_vals_trial[-1] - theta_target_sol
                except Exception:
                    continue

                if abs(error_trial) < abs(error):
                    x, y, psi = x_trial, y_trial, psi_trial
                    print(f"[iter {i}] error = {np.rad2deg(error):.2f}° → {np.rad2deg(error_trial):.2f}°, α={trial_alpha:.4f}")
                    accepted = True
                    break

            if not accepted:
                print(f"[iter {i}] Forcing small fallback step.")
                trial_alpha = 1
                # trial_alpha = 10

                x += trial_alpha * update[1]
                y += trial_alpha * update[2]
                psi += trial_alpha * update[0]
                if not use_position and abs(np.rad2deg(error)) > 10:
                    psi = np.arctan2(x,y)

                x = np.clip(x, 0.01, 0.25)
                y = np.clip(y, 0.01, 0.15)
                psi = psi % (2 * np.pi)

        return best_solution

    flip = theta_target_debug < np.deg2rad(-10)
    theta_target_sol = abs(theta_target_debug)
    first_time = 0
    psi_direct = np.arctan2(0.1, 0.02)
    x_init, y_init, psi_init = 0.1, 0.02, psi_direct
    start_time = time.time()

    if use_position:
        assert target_tip is not None, "Position mode requires target_tip"
        phi = np.arctan2(target_tip[2], target_tip[1])
        x_best, y_best, psi_best, theta_best = run_attempt(
            x_init, y_init, psi_init, first_time, theta_target_sol, start_time, target_tip_3d=target_tip
        )
    else:
        x_best, y_best, psi_best, theta_best = run_attempt(
            x_init, y_init, psi_init, first_time, theta_target_sol, start_time
        )

    if flip:
        y_best = -y_best
        psi_best = (-psi_best + np.pi) % (2 * np.pi) - np.pi

    return x_best, y_best, psi_best, np.rad2deg(theta_best), np.rad2deg(theta_target_debug)

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
    print(f"[INFO] Target bending angle θ = {np.rad2deg(final_bending_rad):.2f}°, direction φ = {np.rad2deg(phi):.2f}°")

    x_solved, y_solved, psi_solved, theta_sol, theta_target_used = inverse_kinematics3(np.deg2rad(final_bending_deg))
    # x_solved, y_solved, psi_solved, theta_sol, theta_target_used = inverse_kinematics(theta_target, target_3d, use_position=True)

    s_vals, theta_vals2, x_vals, y_vals = solve_deflection_angle((x_solved, y_solved), psi_solved)

    path_3d = rotate_2d_to_3d(x_vals, y_vals, phi)
    magnet_pos_3d = rotate_2d_to_3d(np.array([x_solved]), np.array([y_solved]), phi)[0]
    actual_tip = path_3d[-1]
    final_bending_rad2 = theta_vals2[-1]
    final_bending_deg2 = np.rad2deg(final_bending_rad2)
    print(f"Final bending angle θ(L): {final_bending_deg2:.3f} degrees")
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

magnet_pos = res["magnet_pos_3d"]
tip_pos = res["actual_tip"]
psi_pos = res["psi_solved"] 
magnet_pos_tar = res["target_3d"]

print(f"[RESULT] Psi is {psi_pos:.4f} rad = {np.rad2deg(psi_pos):.2f}°")
print(f"[RESULT] Magnet position: x = {magnet_pos[0]:.4f}, y = {magnet_pos[1]:.4f}, z = {magnet_pos[2]:.4f}")
print(f"[RESULT] Final tip position: x = {tip_pos[0]:.4f}, y = {tip_pos[1]:.4f}, z = {tip_pos[2]:.4f}")
print(f"[RESULT] Target position: x = {magnet_pos_tar[0]:.4f}, y = {magnet_pos_tar[1]:.4f}, z = {magnet_pos_tar[2]:.4f}")

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


ax.scatter(*magnet_pos, color='red', s=50, label="Magnet", marker='X')


ax.scatter(*path_3d[0], color='blue', s=60, label='Base', marker='o')


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
