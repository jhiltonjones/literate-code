import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def force_from_paper(r_vec, angle_deg):
    F_sym_eval, T_sym_eval = force_from_paper_sym(sp.Matrix(r_vec), angle_deg)
    F_num = np.array([float(val.evalf()) for val in F_sym_eval])
    T_num = np.array([float(val.evalf()) for val in T_sym_eval])
    return F_num, T_num

def force_from_paper_sym(r_vec, angle_deg):
    mu0 = 4 * sp.pi * 1e-7
    Br = 1
    r, h = 0.04, 0.06
    r_i, h_i = 0.0005, 0.005

    V_E = sp.pi * r**2 * h
    V_I = sp.pi * r_i**2 * h_i

    m_E_mag = (Br * V_E) / mu0
    m_I_mag = (Br * V_I) / mu0

    theta = angle_deg * sp.pi / 180
    m_E = m_E_mag * sp.Matrix([sp.sin(theta), sp.cos(theta), 0])
    m_I = m_I_mag * sp.Matrix([1, 0, 0])

    p = r_vec
    p_norm = sp.sqrt(p.dot(p))
    p_hat = p / p_norm

    Z = sp.eye(3) - 5 * (p_hat * p_hat.T)
    D = sp.eye(3) - 3 * (p_hat * p_hat.T)

    lambda_val = (mu0 / (4 * sp.pi)) * m_E_mag * m_I_mag / (p_norm**3)

    term1 = m_E * m_I.T
    term2 = m_I * m_E.T
    term3 = (m_I.T * Z * m_I)[0] * sp.eye(3)

    F_m = (3 * lambda_val / p_norm) * (term1 + term2 + term3) * p
    T_m = (lambda_val * m_I).cross(D * m_E)

    return F_m, T_m

def inverse_pos_calc(theta_c_desired, initial_guess):
    Ev = 3e6
    Iv = 4.1e-13
    L_total = 0.05
    dtheta_dF = (L_total**2) / (2 * Ev * Iv)
    dtheta_dT = L_total / (Ev * Iv)
    deg2rad = lambda x: x * np.pi / 180

    alpha = 0.1
    tol = 1e-3
    max_iters = 100
    e_log = np.zeros(max_iters)

    # theta_c_desired = deg2rad(64)
    # x_var = np.array([0.01, 0.18, 45.0])
    x_var = initial_guess

    # Symbolic variables
    x, y, angle = sp.symbols('x y angle', real=True)
    r_vec_sym = sp.Matrix([x, y, 0])



    F_sym, T_sym = force_from_paper_sym(r_vec_sym, angle)
    grad_Fx_sym = sp.Matrix([F_sym[0]]).jacobian([x, y, angle])
    grad_Tz_sym = sp.Matrix([T_sym[2]]).jacobian([x, y, angle])

    gradFx_fn = sp.lambdify((x, y, angle), grad_Fx_sym, 'numpy')
    gradTz_fn = sp.lambdify((x, y, angle), grad_Tz_sym, 'numpy')


    for iter in range(max_iters):
        pos_epm = np.array([x_var[0], x_var[1], 0])
        angle_deg = x_var[2]

        F_m, T_m = force_from_paper(pos_epm, angle_deg)
        theta_c_hat = dtheta_dF * F_m[0] + dtheta_dT * (-T_m[2])

        e = theta_c_desired - theta_c_hat
        e_log[iter] = abs(e)

        if abs(e) < tol:
            print(f'Converged in {iter+1} iterations.')
            break

        grad_Fx = np.array(gradFx_fn(x_var[0], x_var[1], x_var[2])).flatten()
        grad_Tz = np.array(gradTz_fn(x_var[0], x_var[1], x_var[2])).flatten()

        J_theta = dtheta_dF * grad_Fx - dtheta_dT * grad_Tz
        delta_x = np.linalg.pinv(J_theta.reshape(-1, 1)) * e
        x_var += alpha * delta_x.flatten()

        x_var[0] = np.clip(x_var[0], -0.05, 0.05)
        x_var[1] = np.clip(x_var[1], 0.17, 0.25)
        x_var[2] = np.clip(x_var[2], 0, 180)

    theta_deg_out = theta_c_desired * 180 / np.pi
    theta_c_desired_deg = np.rad2deg(theta_c_desired)
    print(f'\nFinal magnet pose for θ = {theta_c_desired_deg:.2f} deg:')
    print(f'x = {x_var[0]:.4f} m\ny = {x_var[1]:.4f} m\nangle = {x_var[2]:.2f} deg')

    plt.plot(e_log[:iter])
    plt.xlabel('Iteration')
    plt.ylabel('Bending Angle Error')
    plt.title('Convergence of Bending Angle')
    plt.grid(True)
    # plt.show()
    return theta_deg_out, theta_c_desired, x_var[0], x_var[1], x_var[2]

if __name__ == "__main__":
    x_var = np.array([0, 0.18, 100.0])
    angle = np.deg2rad(-24)
    theta_deg_out, theta_c_desired, x_var[0], x_var[1], x_var[2] = inverse_pos_calc(angle, x_var)
    print(f'Degrees: {x_var[2]} Radians: {np.deg2rad(x_var[2])}')
    print(f'x = {x_var[0]:.4f} m\ny = {x_var[1]:.4f}')