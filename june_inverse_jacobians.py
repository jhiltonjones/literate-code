import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def force_from_paper_sym(r_vec, angle_deg, align_internal=True):
    import sympy as sp

    # Constants
    mu0 = 4 * sp.pi * 1e-7
    Br = 1.4
    r, h = 0.04, 0.06
    r_i, h_i = 0.0005, 0.005

    # Volumes
    V_E = sp.pi * r**2 * h
    V_I = sp.pi * r_i**2 * h_i

    # Dipole magnitudes
    m_E_mag = (Br * V_E) / mu0
    m_I_mag = (Br * V_I) / mu0

    # Dipole directions
    theta = angle_deg * sp.pi / 180

    # External magnet dipole
    m_E = m_E_mag * sp.Matrix([sp.sin(theta), sp.cos(theta), 0])

    # Internal magnet dipole (choose if aligned or fixed)
    if align_internal:
        m_I = m_I_mag * sp.Matrix([sp.sin(theta), sp.cos(theta), 0])
    else:
        m_I = m_I_mag * sp.Matrix([1, 0, 0])  # fixed along x-axis

    # Displacement vector
    p = r_vec
    p_norm2 = p.dot(p)
    p_norm = sp.sqrt(p_norm2)
    p_hat = p / p_norm

    # Interaction tensors
    Z = sp.eye(3) - 5 * (p_hat * p_hat.T)
    D = sp.eye(3) - 3 * (p_hat * p_hat.T)

    # Interaction strength
    lambd = (mu0 / (4 * sp.pi)) * m_E_mag * m_I_mag / (p_norm**3)

    # Magnetic force (you can set this to 0 if focusing on torque only)
    term1 = m_E * m_I.T
    term2 = m_I * m_E.T
    term3 = (m_I.T * Z * m_I)[0] * sp.eye(3)
    F_m = (3 * lambd / p_norm) * (term1 + term2 + term3) * p

    # Magnetic torque
    T_m = lambd * m_I.cross(D * m_E)

    return F_m, T_m



def bend_test(vec, rot):
    Ev = 3e6
    Iv = 4.1e-13
    L_total = 0.03
    dtheta_dF = (L_total**2) / (2 * Ev * Iv)
    dtheta_dT = L_total / (Ev * Iv)
    x, y, angle = sp.symbols('x y angle')
    # r_vec = sp.Matrix([x, y, 0])
    F_sym, T_sym = force_from_paper_sym(vec, rot)

    # Numerical functions
    F_fn = sp.lambdify((x, y, angle), F_sym, modules='numpy')
    T_fn = sp.lambdify((x, y, angle), T_sym, modules='numpy')

    x_, y_, angle_ = x_var
    F_m = F_fn(x_, y_, angle_)
    T_m = T_fn(x_, y_, angle_)
    F_m = np.array(F_m, dtype=float).flatten()
    T_m = np.array(T_m, dtype=float).flatten()
    theta_c_hat = dtheta_dF * F_m[0] + dtheta_dT * (-T_m[2])
    return theta_c_hat

def sympy_solver(angle, x_var, angle2):
    "This is the beam theory that solves the deflection"
    Ev = 3e6
    Iv = 4.1e-13
    L_total = 0.03
    dtheta_dF = (L_total**2) / (2 * Ev * Iv)
    dtheta_dT = L_total / (Ev * Iv)
    deg2rad = lambda x: x * np.pi / 180
    rad2deg = lambda x: x * 180 / np.pi
    initial_guesses = [45, 0, 180]
    alpha = 0.1
    tol = 1e-4
    max_iters = 700
    # Symbolic variables
    x, y, angle = sp.symbols('x y angle')
    r_vec = sp.Matrix([x, y, 0])



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



    e_log = []

    best_error = np.inf
    best_x = None



    for a in initial_guesses:
        x_var = np.array([0.2, 0.03, a], dtype=float)
        print(f"Guess {a}")

        for iter in range(max_iters):
            x_, y_, angle_ = x_var
            F_m = F_fn(x_, y_, angle_)
            T_m = T_fn(x_, y_, angle_)
            F_m = np.array(F_m, dtype=float).flatten()
            T_m = np.array(T_m, dtype=float).flatten()
            theta_c_hat = + dtheta_dT * (-T_m[2])


            e = angle2 - theta_c_hat

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
            x_var[0] = np.clip(x_var[0], 0.15, 0.25)
            x_var[1] = np.clip(x_var[1], 0, 0.04)
            x_var[2] = np.clip(x_var[2], 0, 180)

            # Best solution tracker
            if abs(e) < best_error:
                best_error = abs(e)
                best_x = x_var.copy()


            # if iter % 1000 == 0:
            #     print(f"Iter {iter}: x = {x_var[0]:.3f}, y = {x_var[1]:.3f}, angle = {x_var[2]:.2f}, "
            #           f"θ̂ = {rad2deg(theta_c_hat):.2f}°, error = {abs(e):.4f}")
            #     print(f"‖J‖ = {np.linalg.norm(J_theta):.4e}, ‖Δx‖ = {np.linalg.norm(delta_x):.4e}")
            if iter % 1000 == 0:
                print("Jacobian:", J_theta)
    print(f"\nBest solution found for {np.rad2deg(angle2)}:")
    print(f"x = {best_x[0]:.4f} m\ny = {best_x[1]:.4f} m\nangle = {best_x[2]:.2f} deg")
    print(f"Final error = {rad2deg(best_error):.4f} deg")
    return best_x[0], best_x[1], best_x[2] 

    # # Plotting
    # plt.figure()
    # plt.plot(e_log)
    # plt.xlabel('Iteration')
    # plt.ylabel('Bending Angle Error')
    # plt.title('Convergence of Bending Angle')
    # plt.grid()

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
# Constants


if __name__ == "__main__":
    mu0 = 4 * sp.pi * 1e-7
    Br = 1.4
    r, h = 0.04, 0.06
    r_i, h_i = 0.0005, 0.005
    V_E = sp.pi * r**2 * h
    V_I = sp.pi * r_i**2 * h_i
    m_E_mag = (Br * V_E) / mu0
    m_I_mag = (Br * V_I) / mu0

    # # Parameters
    # Ev = 3e6
    # Iv = 4.1e-13
    # L_total = 0.05
    # dtheta_dF = (L_total**2) / (2 * Ev * Iv)
    # dtheta_dT = L_total / (Ev * Iv)
    # deg2rad = lambda x: x * np.pi / 180
    # rad2deg = lambda x: x * 180 / np.pi

    # alpha = 0.1
    # tol = 1e-4
    # max_iters = 500
    # theta_c_desired = np.deg2rad(20)

    x_var = np.array([0.2, 0.02, 0.0])
    angle = np.deg2rad(44)
    angle2 = angle
    initial_guesses = [45, 0, 180]
    y_calc_pos, x_calc_pos, rotation_calc2 = sympy_solver(angle, x_var, angle2)
    r_vec = sp.Matrix([x_calc_pos,y_calc_pos,0])
    # angle = 30
    F_m, T_m = force_from_paper_sym(r_vec, rotation_calc2)
    # print(f'Force = {F_m} Torque = {T_m}')


    # x_var = np.array([0.17, 0.04, 0.0])
    # # angle = np.deg2rad(24)
    # angle2 = 76
    # x_var[0], x_var[1], x_var[2] = sympy_solver(angle, x_var, angle2)
    # print(f'Degrees: {x_var[2]} Radians: {np.deg2rad(x_var[2])}')
    # print(f'x = {x_var[0]:.4f} m\ny = {x_var[1]:.4f}')

    bending = bend_test(r_vec, rotation_calc2)
    print(f'Bending is {np.rad2deg(bending)}')