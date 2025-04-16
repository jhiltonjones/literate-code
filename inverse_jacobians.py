import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Constants
mu0 = 4 * sp.pi * 1e-7
Br = 1.5
r, h = 0.04, 0.06
r_i, h_i = 0.0005, 0.005
V_E = sp.pi * r**2 * h
V_I = sp.pi * r_i**2 * h_i
m_E_mag = (Br * V_E) / mu0
m_I_mag = (Br * V_I) / mu0

# Symbolic variables
x, y, angle = sp.symbols('x y angle')
r_vec = sp.Matrix([x, y, 0])

def force_from_paper_sym(r_vec, angle_deg):
    theta = angle_deg * sp.pi / 180
    m_E = m_E_mag * sp.Matrix([sp.sin(theta), sp.cos(theta), 0])
    m_I = m_I_mag * sp.Matrix([1, 0, 0])
    p = r_vec
    p_norm2 = p.dot(p)
    p_norm = sp.sqrt(p_norm2)
    p_hat = p / p_norm

    Z = sp.eye(3) - 5 * (p_hat * p_hat.T)
    D = sp.eye(3) - 3 * (p_hat * p_hat.T)

    lambd = (mu0 / (4 * sp.pi)) * m_E_mag * m_I_mag / (p_norm**3)

    term1 = m_E * m_I.T
    term2 = m_I * m_E.T
    term3 = (m_I.T * Z * m_I)[0] * sp.eye(3)
    F_m = (3 * lambd / p_norm) * (term1 + term2 + term3) * 0  # Same as MATLAB *0 (force is zeroed)

    T_m = m_I.cross(D * m_E) * lambd

    return F_m, T_m

# Symbolic Force & Torque
F_sym, T_sym = force_from_paper_sym(r_vec, angle)

# Numerical functions
F_fn = sp.lambdify((x, y, angle), F_sym, modules='numpy')
T_fn = sp.lambdify((x, y, angle), T_sym, modules='numpy')

# Gradients
grad_Fx_sym = sp.derive_by_array(F_sym[0], (x, y, angle))
grad_Tz_sym = sp.derive_by_array(T_sym[2], (x, y, angle))

gradFx_fn = sp.lambdify((x, y, angle), grad_Fx_sym, modules='numpy')
gradTz_fn = sp.lambdify((x, y, angle), grad_Tz_sym, modules='numpy')

# Parameters
Ev = 3e6
Iv = 4.1e-13
L_total = 0.05
dtheta_dF = (L_total**2) / (2 * Ev * Iv)
dtheta_dT = L_total / (Ev * Iv)
deg2rad = lambda x: x * np.pi / 180
rad2deg = lambda x: x * 180 / np.pi

alpha = 0.1
tol = 1e-4
max_iters = 10000
theta_c_desired = deg2rad(-12)

e_log = []

best_error = np.inf
best_x = None

initial_guesses = [45, 0, 180]

for a in initial_guesses:
    x_var = np.array([0.01, 0.18, a], dtype=float)
    print(f"Guess {a}")
    for iter in range(max_iters):
        x_, y_, angle_ = x_var
        F_m = F_fn(x_, y_, angle_)
        T_m = T_fn(x_, y_, angle_)
        F_m = np.array(F_m, dtype=float).flatten()
        T_m = np.array(T_m, dtype=float).flatten()
        theta_c_hat = dtheta_dF * F_m[0] + dtheta_dT * (-T_m[2])


        e = theta_c_desired - theta_c_hat

        e_log.append(abs(e))

        if abs(float(e)) < tol:

            print(f'Converged in {iter} iterations.')
            break

        grad_Fx = gradFx_fn(x_, y_, angle_)
        grad_Tz = gradTz_fn(x_, y_, angle_)
        J_xy = dtheta_dF * np.array(grad_Fx[:2]) - dtheta_dT * np.array(grad_Tz[:2])

        # Finite difference for angle
        eps_angle = 1.0
        angle_up = min(angle_ + eps_angle, 180)
        angle_dn = max(angle_ - eps_angle, 0)

        F_up = F_fn(x_, y_, angle_up)
        T_up = T_fn(x_, y_, angle_up)
        F_dn = F_fn(x_, y_, angle_dn)
        T_dn = T_fn(x_, y_, angle_dn)

        F_up = np.array(F_up, dtype=float).flatten()
        T_up = np.array(T_up, dtype=float).flatten()
        F_dn = np.array(F_dn, dtype=float).flatten()
        T_dn = np.array(T_dn, dtype=float).flatten()


        theta_up = dtheta_dF * F_up[0] + dtheta_dT * (-T_up[2])
        theta_dn = dtheta_dF * F_dn[0] + dtheta_dT * (-T_dn[2])



        dtheta_dangle_fd = (theta_up - theta_dn) / deg2rad(angle_up - angle_dn)
        J_theta = np.hstack([J_xy.flatten(), dtheta_dangle_fd])

        # Inverse Jacobian
        if np.linalg.matrix_rank(J_theta.reshape(1, -1)) == 1:
            inv_J = J_theta[:, np.newaxis] / (np.dot(J_theta, J_theta))
        else:
            inv_J = np.linalg.pinv(J_theta[np.newaxis, :])

        delta_x = inv_J.dot(e).flatten()
        delta_x[2] *= 1000

        # Update
        x_var += alpha * delta_x

        # Clamp
        x_var[0] = np.clip(x_var[0], 0, 0.05)
        x_var[1] = np.clip(x_var[1], 0.17, 0.22)
        x_var[2] = np.clip(x_var[2], 0, 180)

        # Best solution tracker
        if abs(e) < best_error:
            best_error = abs(e)
            best_x = x_var.copy()


        if iter % 1000 == 0:
            print(f"Iter {iter}: x = {x_var[0]:.3f}, y = {x_var[1]:.3f}, angle = {x_var[2]:.2f}, "
                  f"θ̂ = {rad2deg(theta_c_hat):.2f}°, error = {abs(e):.4f}")
            print(f"‖J‖ = {np.linalg.norm(J_theta):.4e}, ‖Δx‖ = {np.linalg.norm(delta_x):.4e}")

    print("\nBest solution found:")
    print(f"x = {best_x[0]:.4f} m\ny = {best_x[1]:.4f} m\nangle = {best_x[2]:.2f} deg")
    print(f"Final error = {rad2deg(best_error):.4f} deg")

# Plotting
plt.figure()
plt.plot(e_log)
plt.xlabel('Iteration')
plt.ylabel('Bending Angle Error')
plt.title('Convergence of Bending Angle')
plt.grid()

# # Plot bending vs angle
# angles = np.linspace(0, 180, 200)
# theta_vals = np.zeros_like(angles)
# for i, angle in enumerate(angles):
#     F = np.array(F, dtype=float).flatten()
#     T = np.array(T, dtype=float).flatten()
#     theta_vals[i] = dtheta_dF * F[0] + dtheta_dT * (-T[2])



# plt.figure()
# plt.plot(angles, rad2deg(theta_vals))
# plt.xlabel('Magnet Angle (deg)')
# plt.ylabel('Estimated Bending θ (deg)')
# plt.title('Bending vs Magnet Rotation')
# plt.grid()
# plt.show()
