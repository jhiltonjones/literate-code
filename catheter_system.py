import control as ct
import numpy as np
import sympy as sp
from force_andtorque import force_from_paper
# === Parameters ===
catheter_params = {
    'Ev': 3e6,
    'Iv': 4.1e-13,
    'L': 0.03,
    'v': 0.00225,
    'Ts': 0.1,
}

def rotate_vector(v, angle_rad):
    """Rotate 2D vector v by angle_rad."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return R @ v

# === Dynamics ===
def catheter_update(t, x, u, params):
    Ev, Iv, L, v, Ts = map(params.get, ['Ev', 'Iv', 'L', 'v', 'Ts'])

    tip = x[0:2]
    base = x[2:4]

    # Step 1: Direction from base to tip
    dir_vec = tip - base
    norm_dir = np.linalg.norm(dir_vec) + 1e-6
    dir_unit = dir_vec / norm_dir

    # Step 2: Estimate tip without bending (for torque calculation)
    tip_est = base + L * dir_unit

    # Step 3: Compute magnetic force/torque based on estimated tip
    r_vec = np.array([u[0] - tip_est[0], u[1] - tip_est[1], 0])
    angle = u[2]
    F_m, T_m = force_from_paper(r_vec, angle)
    Fy = F_m[0]
    Tz = -T_m[2]

    # Step 4: Compute bending amount (gamma)
    gamma_c_hat = (Fy * L**3) / (3 * Ev * Iv) + (Tz * L**2) / (2 * Ev * Iv)
    theta_c_hat = (Fy * L**2) / (2 * Ev * Iv) + (Tz * L) / (Ev * Iv)

    # Step 5: Get bending direction (based on torque)
    bending_dir = np.cross(T_m, np.array([0, 0, 1]))[:2]
    if np.linalg.norm(bending_dir) < 1e-6:
        bending_dir = np.array([-dir_unit[1], dir_unit[0]])  # fallback
    bending_dir /= np.linalg.norm(bending_dir)

    # Apply angular deflection to current direction
    new_dir = rotate_vector(dir_unit, theta_c_hat)

    # Now advance base and compute tip
    base_next = base + v * Ts * new_dir
    tip_next = base_next + L * new_dir


    # Final state
    state_next = np.concatenate((tip_next, base_next))

    if params.get("return_full", False):
        return state_next, gamma_c_hat, theta_c_hat
    else:
        return state_next


# === Output function ===
def catheter_output(t, x, u, params):
    return x

# === Build nonlinear system ===
catheter_sys = ct.nlsys(
    catheter_update, catheter_output,
    inputs=['magnet_x', 'magnet_y', 'magnet_angle'],
    states=['tip_x', 'tip_y', 'base_x', 'base_y'],
    outputs=['tip_x', 'tip_y', 'base_x', 'base_y'],
    params=catheter_params,
    dt=catheter_params['Ts'],
    name='catheterMPC'
)
