import math
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Parameters for MSCR #1 (length, radius, modulus, magnetization)
L = 0.05  # rod length in meters (24 mm)
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
print(M)
MAGNET_M = 318           # magnet's dipole moment magnitude (A·m^2), calibrated for the N52 magnet
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

def total_finite_difference(theta_val, x_val, y_val, x_m, y_m, psi_val, h=1e-6):
    dx_dtheta = -np.sin(theta_val)
    dy_dtheta =  np.cos(theta_val)

    x_f = x_val + h * dx_dtheta
    y_f = y_val + h * dy_dtheta
    x_b = x_val - h * dx_dtheta
    y_b = y_val - h * dy_dtheta

    # Forward rotation of internal dipole
    theta_f = theta_val + h
    theta_b = theta_val - h

    m_f = np.array([np.cos(theta_f), np.sin(theta_f)])
    m_b = np.array([np.cos(theta_b), np.sin(theta_b)])

    Bx_f, By_f = magnetic_field((x_f, y_f), (x_m, y_m), psi_val)
    Bx_b, By_b = magnetic_field((x_b, y_b), (x_m, y_m), psi_val)

    B_f = np.array([Bx_f, By_f])
    B_b = np.array([Bx_b, By_b])

    dot_f = m_f @ B_f
    dot_b = m_b @ B_b

    return (dot_f - dot_b) / (2 * h)

def magnetic_field(point, magnet_pos, magnet_dipole_angle):
    """
    Compute magnetic field (Bx, By) at a given 2D point due to the dipole magnet.
    Uses the point dipole model: 
      b = (μ0 * M_A / (4π * r^3)) * [3(hat p · hat m)hat p - hat m],
    where hat p is the unit vector from magnet to point, and hat m is unit dipole orientation.
    """
    x, y = point
    x_m, y_m = magnet_pos
    # Vector from magnet to point
    px = x - x_m
    py = y - y_m
    r_sq = px*px + py*py
    r_mag = math.sqrt(r_sq)
    if r_mag == 0:  # avoid singularity if point == magnet position
        return (0.0, 0.0)
    # Unit vector from magnet to point
    phat_x = px / r_mag
    phat_y = py / r_mag
    # Magnet's dipole unit vector (orientation)
    mhat_x = math.cos(magnet_dipole_angle)
    mhat_y = math.sin(magnet_dipole_angle)
    # Compute magnetic field components via point dipole model
    c = phat_x*mhat_x + phat_y*mhat_y  # dot product (\hat p · \hat m)
    factor = MU0 * MAGNET_M / (4 * math.pi * (r_mag**3))
    Bx = factor * (3 * c * phat_x - mhat_x)
    By = factor * (3 * c * phat_y - mhat_y)
    return (Bx, By)
def finite_difference_second_term(theta_val, x_val, y_val, x_m, y_m, psi_val, h=1e-6):
    # Derivative of position w.r.t. theta
    dx_dtheta = -np.sin(theta_val)
    dy_dtheta = np.cos(theta_val)

    # Forward and backward positions
    x_forward = x_val + h * dx_dtheta
    y_forward = y_val + h * dy_dtheta
    x_backward = x_val - h * dx_dtheta
    y_backward = y_val - h * dy_dtheta

    # B at forward and backward
    Bx_f, By_f = magnetic_field((x_forward, y_forward), (x_m, y_m), psi_val)
    Bx_b, By_b = magnetic_field((x_backward, y_backward), (x_m, y_m), psi_val)

    # Internal dipole direction
    m = np.array([np.cos(theta_val), np.sin(theta_val)])

    B_f = np.array([Bx_f, By_f])
    B_b = np.array([Bx_b, By_b])

    # Central difference: d/dtheta (m . B(x(theta)))
    dB_dtheta = (m @ (B_f - B_b)) / (2 * h)
    return dB_dtheta

# --- Step 1: Rebuild symbolic second term ---
# Redefine symbolic variables
theta, x, y, x_m, y_m, psi = sp.symbols('theta x y x_m y_m psi')
C = sp.Symbol('C')
C_grad = sp.Symbol('C_grad')
px = x - x_m
py = y - y_m
r = sp.sqrt(px**2 + py**2)
a = px / r
b = py / r

# Define external magnet unit vector
m_hat = sp.Matrix([sp.cos(psi), sp.sin(psi)])
p_hat = sp.Matrix([a, b])

# Magnetic field vector b
dot_pm = p_hat.dot(m_hat)
b_vec = C * (3 * dot_pm * p_hat - m_hat)



# Internal rotated dipole and dx/dtheta
# Re-define for clarity
theta = sp.Symbol('theta')
Rm = sp.Matrix([sp.cos(theta), sp.sin(theta)])
Rm_dtheta = sp.Matrix([-sp.sin(theta), sp.cos(theta)])

# dot_product = Rm.dot(b_vec)
# dot_derivative = sp.simplify(sp.diff(dot_product, theta))
# manual_expected = sp.simplify(Rm_dtheta.dot(b_vec))

# # Check equality

dx_dtheta = sp.Matrix([-sp.sin(theta), sp.cos(theta)])

# Gradient of b with respect to x, y
grad_b = sp.Matrix([[sp.diff(bi, x), sp.diff(bi, y)] for bi in b_vec])

# Second term of chain rule
first_term_expr = sp.simplify(sp.diff(Rm.dot(b_vec), theta))
# print(f"First term is : {first_term_expr}")
# assert first_term_expr.equals(manual_expected), "Derivative does not match manual expected value"


f_first_term = sp.lambdify(
    (theta, x, y, x_m, y_m, psi, C),
    first_term_expr,
    modules="numpy"
)


m1 = sp.cos(psi)
m2 = sp.sin(psi)
a_sym = px / r
b_sym = py / r

# Scalar dot product
dot_pm = a_sym * m1 + b_sym * m2

# Outer products (manual scalars)
ppT = sp.Matrix([[a_sym * a_sym, a_sym * b_sym],
                 [b_sym * a_sym, b_sym * b_sym]])

pmT = sp.Matrix([[a_sym * m1, a_sym * m2],
                 [b_sym * m1, b_sym * m2]])

mpT = sp.Matrix([[m1 * a_sym, m1 * b_sym],
                 [m2 * a_sym, m2 * b_sym]])

I2 = sp.eye(2)
Z = I2 - 5 * ppT

grad_b_explicit = C_grad * (
    pmT + dot_pm * I2 + Z * mpT
)

Rm = sp.Matrix([sp.cos(theta), sp.sin(theta)])
dx_dtheta = sp.Matrix([-sp.sin(theta), sp.cos(theta)])

# Second term
second_term_expr_explicit = sp.simplify((grad_b_explicit.T * Rm).dot(dx_dtheta))

# Lambdify
f_second_term = sp.lambdify(
    (theta, x, y, x_m, y_m, psi, C_grad),
    second_term_expr_explicit,
    modules='numpy'
)
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
    fd_second_term = finite_difference_second_term(theta_val, x_val, y_val, x_m, y_m, magnet_dipole_angle)
    # print("Finite-difference second term:", fd_second_term)
    
    # Total derivative

    total_symbolic = first_term + second_term
    fd_total = total_finite_difference(theta_val, x_val, y_val, x_m, y_m, magnet_dipole_angle)
    # print("Total symbolic derivative:", total_symbolic)
    # print("Total finite difference:", fd_total)
    # print("Absolute error:", abs(total_symbolic - fd_total))

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
        if attempts > 100:
            print("[WARNING] No sign change in integrate_curvature. Switching to fallback k0 range.")
            # Try the opposite direction
            k0_low, k0_high = -50, 50
            res_low = integrate_curvature(k0_low)
            res_high = integrate_curvature(k0_high)
            break

        res_high = integrate_curvature(k0_high)
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

def compute_jacobian(s_vals, theta_vals, x_vals, y_vals, magnet_pos, magnet_dipole_angle):
    """
    Compute the analytical Jacobian J(s) = ∂θ(s)/∂ψ along the rod.
    Solves the linear variational ODE for J(s) and applies superposition to enforce J'(L)=0 
    if Assumption 2 (Lipschitz constant K sufficiently small) holds; otherwise uses the particular solution.
    """
    n = len(s_vals)
    # Precompute coefficient a(s) = ∂f/∂θ and forcing term b(s) = ∂f/∂ψ along θ(s)
    a_vals = [0.0] * n
    b_vals = [0.0] * n
    cos_vals = [math.cos(theta) for theta in theta_vals]
    sin_vals = [math.sin(theta) for theta in theta_vals]
    for i in range(n):
        # Magnetic field at this point on the rod
        Bx, By = magnetic_field((x_vals[i], y_vals[i]), magnet_pos, magnet_dipole_angle)
        # ∂f/∂θ = (A*M/(E*I)) * (Bx * cosθ + By * sinθ)
        a_vals[i] = (A * M / (E * I)) * (Bx * cos_vals[i] + By * sin_vals[i])
        # Compute ∂Bx/∂ψ and ∂By/∂ψ (treating position fixed, i.e., ignoring higher-order ∇^2 b terms)
        px = x_vals[i] - magnet_pos[0]
        py = y_vals[i] - magnet_pos[1]
        r_sq = px*px + py*py
        if r_sq == 0:
            dBx_dψ = dBy_dψ = 0.0
        else:
            r_mag = math.sqrt(r_sq)
            a_hat = px / r_mag
            b_hat = py / r_mag
            cosψ = math.cos(magnet_dipole_angle)
            sinψ = math.sin(magnet_dipole_angle)
            factor = MU0 * MAGNET_M / (4 * math.pi * (r_mag**3))
            # Partial derivatives of dipole field w.rt ψ (in-plane rotation)
            dBx_dψ = factor * ((1 - 3 * a_hat * a_hat) * sinψ + 3 * a_hat * b_hat * cosψ)
            dBy_dψ = factor * (((3 * b_hat * b_hat) - 1) * cosψ - 3 * a_hat * b_hat * sinψ)
        # ∂f/∂ψ = (A*M/(E*I)) * [dBx/dψ * sinθ - dBy/dψ * cosθ]
        b_vals[i] = (A * M / (E * I)) * (dBx_dψ * sin_vals[i] - dBy_dψ * cos_vals[i])

    # Solve J''(s) = a(s) * J(s) + b(s) with J(0)=0.
    # Use superposition: J = J_p + C * J_h, where J_p is particular solution (J(0)=0, J'(0)=0),
    # and J_h is homogeneous solution (J(0)=0, J'(0)=1).
    J_p = [0.0] * n
    J_h = [0.0] * n
    dJ_p = 0.0  # J'_p(0)
    dJ_h = 1.0  # J'_h(0)
    for i in range(n - 1):
        ds = s_vals[i+1] - s_vals[i]
        # Integrate Jacobian ODE using Euler steps
        # J''_p = a(s)*J_p + b(s);  J''_h = a(s)*J_h
        ddJ_p = a_vals[i] * J_p[i] + b_vals[i]
        ddJ_h = a_vals[i] * J_h[i]
        J_p[i+1] = J_p[i] + dJ_p * ds
        dJ_p    += ddJ_p * ds
        J_h[i+1] = J_h[i] + dJ_h * ds
        dJ_h    += ddJ_h * ds

    # Enforce boundary condition J'(L)=0 if Lipschitz condition (Assumption 2) is satisfied
    J_vals = J_p  # default to particular solution
    # Estimate Lipschitz constant K = max|∂f/∂θ| and check Assumption 2 (e.g., K * L < π)
    if n > 0:
        K_est = max(abs(a) for a in a_vals)
        if K_est * L < math.pi:
            # Compute combination factor C such that J'(L)=0:  C = -J'_p(L) / J'_h(L)
            C = - dJ_p / dJ_h if dJ_h != 0 else 0.0
            J_vals = [J_p[i] + C * J_h[i] for i in range(n)]
    return J_vals

def rotate_vector(v, axis, angle_rad):
    """
    Rotate vector v around the given axis by angle_rad using Rodrigues' rotation formula.
    """
    axis = axis / np.linalg.norm(axis)
    v_rot = (v * np.cos(angle_rad) +
             np.cross(axis, v) * np.sin(angle_rad) +
             axis * np.dot(axis, v) * (1 - np.cos(angle_rad)))
    return v_rot

# Original 3D magnet position
magnet_3d = np.array([0.1, 0.1, 0.06])  # x, y, z
catheter_base = np.array([0.05, 0.0, 0.0])

phi = compute_phi_from_yz(magnet_3d[1], magnet_3d[2])

# Step 2: Project 3D magnet into 2D solver plane
magnet_2d = project_magnet_to_2d(magnet_3d[0], magnet_3d[1], magnet_3d[2])

# Step 3: Solve in 2D
# psi = np.arctan2(magnet_2d[1] - 0.0, magnet_2d[0] - 0.0)
delta = np.deg2rad(180)  # or any desired angle in radians

psi = np.arctan2(magnet_2d[1], magnet_2d[0]) + delta

s_vals, theta_vals, x_vals, y_vals = solve_deflection_angle(magnet_2d, psi)

# Step 4: Rotate the 2D result back into 3D
path_3d = rotate_2d_to_3d(x_vals, y_vals, phi)

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
# rho_G = 0.18  # 25 cm distance
# phi = math.radians(90)  # magnet position angle (90° from rod axis, i.e. directly above base)
# magnet_position = (rho_G * math.cos(phi), rho_G * math.sin(phi))
# magnet_position = (pos1, pos2)
# print(f"Magnet position: {magnet_position}")
# catheter_pos = np.array([0.05, 0])  # Catheter starts at origin

# # Direction vector from catheter to magnet
# direction = magnet_position - catheter_pos

# # Compute angle in radians and degrees
# angle_rad = np.arctan2(direction[1], direction[0])
# angle_deg = np.rad2deg(angle_rad)

# reverse_angle = -1* angle_deg

# angle_custom = np.deg2rad(-40)
# # psi = np.deg2rad(reverse_angle)
# # psi = angle_custom
# psi = angle_rad
# print(f"Angle of EPM rotation is: {np.rad2deg(psi)}")
# psi = math.radians(90)  # magnet's dipole orientation (clockwise 90° from pointing along +x, so pointing downward)
# Compute deflection and Jacobian
# s_vals, theta_vals, x_vals, y_vals = solve_deflection_angle(magnet_position, psi)
# J_vals = compute_jacobian(s_vals, theta_vals, x_vals, y_vals, magnet_position, psi)
# # phi = np.deg2rad(45)  # radians

# # # Rotate into 3D
# # path_3d = rotate_2d_path_to_3d(x_vals, y_vals, phi, axis='x')
# # Output the tip deflection angle and Jacobian for demonstration
# tip_angle = theta_vals[-1]
# tip_jacobian = J_vals[-1]
# print(f"Tip deflection angle θ(L) = {np.rad2deg(tip_angle):.4f} rad")
# print(f"Analytical Jacobian J(L) = {tip_jacobian:.4f} (∂θ(L)/∂ψ in rad⁻¹)")

# def rotate_2d_path_to_3d(x_vals, y_vals, phi, axis='x'):
#     """
#     Rotate 2D x-y curve into 3D by rotating around specified axis.
#     """
#     x_vals = np.array(x_vals)
#     y_vals = np.array(y_vals)
#     if axis == 'x':
#         y_rot = y_vals * np.cos(phi)
#         z_vals = y_vals * np.sin(phi)
#         return np.vstack([x_vals, y_rot, z_vals]).T
#     else:
#         raise NotImplementedError("Only rotation around x-axis supported.")

# magnet_position_3d = np.array([pos1, pos2, 0.0])  # (x, y, z)
# catheter_base = np.array([0.05, 0, 0])

# # Compute vector and angles
# direction = magnet_position_3d - catheter_base
# psi = np.arctan2(direction[1], direction[0])     # Dipole angle in xy-plane
# phi = np.arctan2(direction[2], direction[1])     # Rotation angle in yz-plane

# path_3d = rotate_2d_path_to_3d(x_vals, y_vals, phi, axis='x')

# # -- Plot --

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.plot(path_3d[:, 0], path_3d[:, 1], path_3d[:, 2], label="3D Deflection")
# ax.scatter(*magnet_position_3d, color='red', label='Magnet')
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.legend()
# plt.tight_layout()
# plt.show()

# import matplotlib.pyplot as plt

# plt.figure(figsize=(6, 6))

# # Plot catheter deformation
# plt.plot(x_vals, y_vals, linewidth=2, label=f'ψ = {math.degrees(psi):.1f}°')

# # Plot magnet position
# plt.scatter(magnet_position[0], magnet_position[1], color='red', s=80, label='Magnet')
# plt.text(magnet_position[0], magnet_position[1], '  Magnet', color='red', fontsize=10, va='bottom')

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
#     J_vals = compute_jacobian(s_vals, theta_vals, x_vals, y_vals, magnet_position, psi)

#     tip_angle = theta_vals[-1]
#     tip_jacobian = J_vals[-1]

#     print(f"  Tip deflection angle θ(L) = {np.rad2deg(tip_angle):.2f}°")
#     print(f"  Jacobian ∂θ(L)/∂ψ = {tip_jacobian:.4f} rad⁻¹")

#     # Plot the catheter deformation for this ψ
#     plt.plot(x_vals, y_vals, label=f'ψ = {deg}°')

# # Plot magnet position
# plt.scatter(magnet_position[0], magnet_position[1], color='red', s=80, label='Magnet')
# plt.text(magnet_position[0], magnet_position[1], '  Magnet', color='red', fontsize=10, va='bottom')

# # Final plot styling
# plt.xlabel('x (m)')
# plt.ylabel('y (m)')
# plt.title('MSCR Deformation for Various Magnet Orientations')
# plt.axis('equal')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()






# import numpy as np
# import matplotlib.pyplot as plt

# # Define the test point near the tip
# x_test = x_vals[-1]
# y_test = y_vals[-1]

# # # Fixed magnet parameters
# # magnet_pos = magnet_position
# # psi_val = psi

# # # Compute magnetic field at the tip
# # Bx_fixed, By_fixed = magnetic_field((x_test, y_test), magnet_pos, psi_val)
# # B_fixed = np.array([Bx_fixed, By_fixed])

# # # Define theta range and compute dot product and its derivative
# theta_range = np.linspace(0, np.pi, 300)
# # dot_product_vals = []
# # torque_derivative_vals = []

# # for theta_val in theta_range:
# #     m = np.array([np.cos(theta_val), np.sin(theta_val)])
# #     dm_dtheta = np.array([-np.sin(theta_val), np.cos(theta_val)])
# #     dot_product_vals.append(m @ B_fixed)
# #     torque_derivative_vals.append(dm_dtheta @ B_fixed)

# # # Plot both dot product and its derivative (torque term)
# # plt.figure(figsize=(8, 5))
# # plt.plot(np.rad2deg(theta_range), dot_product_vals, label=r'$m(\theta) \cdot B$', color='blue')
# # plt.plot(np.rad2deg(theta_range), torque_derivative_vals, label=r'$\frac{d}{d\theta}(m \cdot B)$ (torque)', color='red', linestyle='--')
# # plt.axvline(np.rad2deg(theta_vals[-1]), color='gray', linestyle=':', label=r'Tip $\theta(L)$')
# # plt.xlabel(r'$\theta$ [deg]')
# # plt.ylabel('Value')
# # plt.title('Dot Product and Torque vs. Internal Dipole Angle')
# # plt.legend()
# # plt.grid(True)
# # plt.tight_layout()
# # plt.show()

# # Try different dipole angles and plot comparison

# # Store tip angles for each psi
# # psi_degrees_list = [60, 75, 90]
# # theta_tip_vals = []

# # plt.figure(figsize=(8, 5))

# # for psi_deg in psi_degrees_list:
# #     psi_val = math.radians(psi_deg)
    
# #     # Solve for current psi
# #     s_vals, theta_vals, x_vals, y_vals = solve_deflection_angle(magnet_position, psi_val)
# #     theta_tip_vals.append(theta_vals[-1])

# #     # Compute torque derivative curve at tip
# #     Bx_fixed, By_fixed = magnetic_field((x_vals[-1], y_vals[-1]), magnet_position, psi_val)
# #     B_fixed = np.array([Bx_fixed, By_fixed])
# #     torque_derivative_vals = [
# #         np.array([-np.sin(th), np.cos(th)]) @ B_fixed for th in theta_range
# #     ]

# #     # Plot torque derivative curve
# #     plt.plot(np.rad2deg(theta_range), torque_derivative_vals, label=f'ψ = {psi_deg}°')

# # # Add vertical lines at each tip angle
# # for psi_deg, theta_tip in zip(psi_degrees_list, theta_tip_vals):
# #     plt.axvline(np.rad2deg(theta_tip), linestyle=':', linewidth=1.5, label=f'θ(L) for ψ={psi_deg}°')

# # # Final plot labels
# # plt.xlabel(r'$\theta$ [deg]')
# # plt.ylabel(r'$\frac{d}{d\theta}(m \cdot B)$')
# # plt.title('Torque Derivative with Tip Angles for Each ψ')
# # plt.grid(True)
# # plt.legend()
# # plt.tight_layout()
# # # plt.show()

# def update_psi_to_target_theta_fast(theta_desired_deg, initial_psi_deg, magnet_position, max_iter=25, tol=1e-3):
#     psi_deg = initial_psi_deg
#     theta_desired_rad = math.radians(theta_desired_deg)
#     prev_error = None

#     for i in range(max_iter):
#         psi_rad = math.radians(psi_deg)

#         # Low-resolution solve to save time
#         s_vals, theta_vals, x_vals, y_vals = solve_deflection_angle(magnet_position, psi_rad, n_steps=250)
#         theta_tip = theta_vals[-1]
#         J_vals = compute_jacobian(s_vals, theta_vals, x_vals, y_vals, magnet_position, psi_rad)
#         J_theta = -1*J_vals[-1]

#         error = theta_desired_rad - theta_tip

#         if abs(J_theta) < 1e-6:
#             print("[Warning] Jacobian too small to proceed.")
#             break

#         # Adaptive step with clamping
#         step = np.clip(5.0 * error / J_theta, -5.0, 5.0)
#         psi_deg += step

#         print(f"[Step {i}] ψ = {psi_deg:.2f}°, θ(L) = {np.degrees(theta_tip):.2f}°, error = {np.degrees(error):.3f}°")

#         if abs(error) < math.radians(tol):
#             break

#         if prev_error is not None and abs(error) > abs(prev_error):
#             print("[Warning] Error increasing — exiting.")
#             break
#         prev_error = error

#     return psi_deg

# # Modify the original function to target a desired bending angle theta(L) instead of position
# def update_psi_to_target_theta(theta_desired_deg, initial_psi_deg, magnet_position, max_iter=25, tol=1e-4):
#     psi_deg = initial_psi_deg
#     theta_desired_rad = math.radians(theta_desired_deg)

#     for i in range(max_iter):
#         psi_rad = math.radians(psi_deg)

#         # Solve for deflection
#         s_vals, theta_vals, x_vals, y_vals = solve_deflection_angle(magnet_position, psi_rad)
#         theta_tip = theta_vals[-1]
#         J_vals = compute_jacobian(s_vals, theta_vals, x_vals, y_vals, magnet_position, psi_rad)
#         J_theta = J_vals[-1]  # ∂θ(L)/∂ψ

#         # Compute error
#         error = theta_desired_rad - theta_tip

#         if abs(J_theta) < 1e-8:
#             print("[Warning] Jacobian is too small to make progress.")
#             break

#         # Update rule using gradient descent
#         alpha = 5.0  # tuning parameter
#         step = alpha * error / J_theta
#         psi_deg -= step

#         print(f"[Step {i}] ψ = {psi_deg:.2f}°, θ(L) = {np.degrees(theta_tip):.2f}°, error = {np.degrees(error):.4f}°")

#         if abs(error) < math.radians(tol):
#             break

#     return psi_deg

# # Example usage
# theta_desired_deg = 20
# initial_guess_deg = 45
# psi_opt = update_psi_to_target_theta_fast(theta_desired_deg, initial_guess_deg, magnet_position)

# # Solve and visualize
# final_psi_rad = math.radians(psi_opt)
# s_vals, theta_vals, x_vals, y_vals = solve_deflection_angle(magnet_position, final_psi_rad)

# plt.figure(figsize=(6, 6))
# plt.plot(x_vals, y_vals, 'b-', label=f'ψ = {psi_opt:.1f}°')
# plt.axhline(0, color='gray', linestyle='--')
# plt.xlabel("x (m)")
# plt.ylabel("y (m)")
# plt.title(f"MSCR Shape for Target θ(L) = {theta_desired_deg}°")
# plt.axis("equal")
# plt.grid(True)
# plt.legend()
# plt.show()




# # from scipy.integrate import quad
# # from scipy.interpolate import interp1d
# # from scipy.integrate import solve_ivp
# # from scipy.optimize import root

# # # --- Convert data to interpolable form ---
# # s_plot = np.array(s_vals)
# # theta_interp = interp1d(s_plot, theta_vals, kind='cubic')
# # J_interp = interp1d(s_plot, J_vals, kind='cubic')

# # def theta_of_s(s): return theta_interp(s)
# # def J_of_s(s): return J_interp(s)

# # def dxL_dpsi_integrand(s): return np.cos(theta_of_s(s)) * J_of_s(s)
# # def dyL_dpsi_integrand(s): return np.sin(theta_of_s(s)) * J_of_s(s)

# # def compute_control_input(xL_des, xL_dot_des, theta_func, s_plot, J_vals, psi_deg):
# #     dxL_dpsi, _ = quad(dxL_dpsi_integrand, 0, L)
# #     dyL_dpsi, _ = quad(dyL_dpsi_integrand, 0, L)

# #     dxL_dnu = 1.0
# #     dyL_dnu = 0.0
# #     dzL_dnu = 0.0
# #     dzL_dpsi = 0.0

# #     B = np.array([
# #         [dxL_dpsi],
# #         [dyL_dpsi]
# #     ])
# #     xL = np.array([x_vals[-1], y_vals[-1], 0.0])  # tip position from shape

# #     kx = 0.5
# #     B_pinv = np.linalg.pinv(B)
# #     u = B_pinv @ (xL_dot_des + kx * (xL_des - xL))
# #     nu_dot, psi_dot_rad = u

# #     psi_dot_deg = np.rad2deg(psi_dot_rad)
# #     psi_deg_new = np.clip(psi_deg + psi_dot_deg, 0, 90)  # restrict to [0°, 90°]


# #     print(f"Computed control input: ν̇ = {nu_dot:.6f}, ψ̇ = {psi_dot_deg:.2f} deg → new ψ = {psi_deg_new:.2f} deg")
# #     print(f"xL current = {xL}")
# #     print(f"xL desired = {xL_des}")
# #     print(f"Error = {xL_des - xL}")
# #     print(f"B matrix =\n{B}")
# #     print(f"B⁺ (pseudo-inverse) =\n{B_pinv}")

# #     return nu_dot, psi_deg_new

# # def solve_catheter_shape_for_psi(psi_deg):
# #     psi_rad = math.radians(psi_deg)
# #     magnet_position = (rho_G * math.cos(psi_rad), rho_G * math.sin(psi_rad))
# #     _, _, x_vals_new, y_vals_new = solve_deflection_angle(magnet_position, psi_rad)
# #     return x_vals_new, y_vals_new

# # def visualize_catheter_shapes(x0, y0, x1, y1, xL_des, psi0, psi1):
# #     xL = np.array([x1[-1], y1[-1]])
# #     plt.figure(figsize=(6, 6))
# #     plt.plot(x0, y0, 'b-', label=f'Initial (ψ={psi0:.1f}°)')
# #     plt.plot(x1, y1, 'g--', label=f'Updated (ψ={psi1:.1f}°)')
# #     plt.plot(xL_des[0], xL_des[1], 'rx', markersize=10, label='Target Tip')
# #     plt.quiver(x1[-1], y1[-1], xL_des[0] - x1[-1], xL_des[1] - y1[-1], 
# #                angles='xy', scale_units='xy', scale=1, color='gray', width=0.005, label='Error Vector')
# #     plt.xlabel("X (m)")
# #     plt.ylabel("Y (m)")
# #     plt.title("Catheter Shape: Before and After Control")
# #     plt.grid(True)
# #     plt.axis("equal")
# #     plt.legend()
# #     plt.tight_layout()
# #     plt.show()
# # def compute_dtheta_dxL(theta_interp, J_interp):
# #     def dxL_dpsi_integrand(s): return np.cos(theta_interp(s)) * J_interp(s)
# #     def dyL_dpsi_integrand(s): return np.sin(theta_interp(s)) * J_interp(s)
# #     dxL_dpsi, _ = quad(dxL_dpsi_integrand, 0, L)
# #     dyL_dpsi, _ = quad(dyL_dpsi_integrand, 0, L)
# #     return dxL_dpsi, dyL_dpsi

# # def control_loop_to_target(xL_des, psi_deg_initial, steps=10, kx=0.5):
# #     psi_deg = psi_deg_initial
# #     history = []

# #     for step in range(steps):
# #         # Update shape and Jacobian
# #         psi_rad = math.radians(psi_deg)
# #         # magnet_position = (rho_G * math.cos(psi_rad), rho_G * math.sin(psi_rad))
# #         s_vals, theta_vals, x_vals, y_vals = solve_deflection_angle(magnet_position, psi_rad)
# #         J_vals = compute_jacobian(s_vals, theta_vals, x_vals, y_vals, magnet_position, psi_rad)
# #         theta_interp = interp1d(s_vals, theta_vals, kind='cubic')
# #         J_interp = interp1d(s_vals, J_vals, kind='cubic')

# #         # Compute Jacobian components for dxL/dψ and dyL/dψ
# #         dxL_dpsi, dyL_dpsi = compute_dtheta_dxL(theta_interp, J_interp)

# #         # Tip position
# #         xL = np.array([x_vals[-1], y_vals[-1]])
# #         error = xL_des[:2] - xL

# #         # 2D Jacobian
# #         B = np.array([[dxL_dpsi],
# #                       [dyL_dpsi]])

# #         # Control law (simple P controller)
# #         try:
# #             B_pinv = np.linalg.pinv(B)
# #             psi_dot_rad = (B_pinv @ (kx * error))[0]
# #         except np.linalg.LinAlgError:
# #             print("Jacobian inversion failed, skipping update.")
# #             break

# #         # Update ψ
# #         psi_deg -= math.degrees(psi_dot_rad)
# #         psi_deg = np.clip(psi_deg, 0, 360)

# #         # Log for visualization
# #         history.append((x_vals, y_vals, psi_deg, np.linalg.norm(error)))

# #         print(f"Step {step+1}: ψ = {psi_deg:.2f}°, error = {error}, |error| = {np.linalg.norm(error):.6f}")
# #         print(f"J = [{dxL_dpsi:.4f}, {dyL_dpsi:.4f}], error = {error}")
# #         print(f"Projection = {dxL_dpsi * error[0] + dyL_dpsi * error[1]}")

# #         # Early stop if error is very small
# #         if np.linalg.norm(error) < 1e-4:
# #             break

# #     return history

# # initial_psi_deg = psi
# # target_tip = np.array([0.02, 0.005])  # Your target tip position (2cm, 0.5cm)
# # history = control_loop_to_target(target_tip, initial_psi_deg, steps=15)

# # # Visualize first and final catheter shape
# # x0, y0, psi0, _ = history[0]
# # x1, y1, psi1, _ = history[-1]

# # visualize_catheter_shapes(x0, y0, x1, y1, target_tip, psi0, psi1)

# from numpy.linalg import lstsq

# # --- Fit sinusoidal model to θ(L) as a function of ψ ---

# def compute_theta_L_for_psi(psi_rad):
#     # magnet_position = (rho_G * math.cos(psi_rad), rho_G * math.sin(psi_rad))
#     s_vals, theta_vals, _, _ = solve_deflection_angle(magnet_position, psi_rad)
#     return theta_vals[-1]  # tip angle θ(L)

# # Sweep psi from 0 to 360 degrees
# psi_vals = np.linspace(0, 2*np.pi, 36)  # more resolution = smoother fit
# theta_L_vals = np.array([compute_theta_L_for_psi(psi_i) for psi_i in psi_vals])

# # Design matrix for bias-free sinusoidal model: [ -cos(ψ), sin(ψ) ]
# X = np.vstack([
#     -np.cos(psi_vals),
#     np.sin(psi_vals)
# ]).T

# # Least squares solve (no bias)
# coeffs, _, _, _ = lstsq(X, theta_L_vals, rcond=None)
# vartheta_1, vartheta_2 = coeffs

# # Reconstruct fit
# theta_L_fit = X @ coeffs
# vartheta = np.sqrt(vartheta_1**2 + vartheta_2**2)
# psi_0 = np.arctan2(vartheta_1, vartheta_2)

# # Print results
# print(f"Estimated ϑ̄₁ = {vartheta_1:.4f}, ϑ̄₂ = {vartheta_2:.4f}")
# print(f"Amplitude ϑ̄ = {vartheta:.4f}")
# print(f"Phase shift ψ₀ = {np.rad2deg(psi_0):.2f} deg")

# # Plot the result
# plt.figure(figsize=(6, 4))
# plt.plot(np.rad2deg(psi_vals), np.rad2deg(theta_L_vals), label="θ_L (simulated)")
# plt.plot(np.rad2deg(psi_vals), np.rad2deg(theta_L_fit), '--', label="Fitted sinusoid (no bias)")
# plt.xlabel("ψ (deg)")
# plt.ylabel("θ(L) (deg)")
# plt.title("Bias-Free Tip Deflection θ(L) vs Magnet Rotation ψ")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

