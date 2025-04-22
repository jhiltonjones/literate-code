import torch
import numpy as np
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)



def force_from_paper_torch(x, y, angle_deg, requires_grad=True):
    if requires_grad:
        x = torch.tensor(x, dtype=torch.get_default_dtype(), requires_grad=True)
        y = torch.tensor(y, dtype=torch.get_default_dtype(), requires_grad=True)
        theta_deg = torch.tensor(angle_deg, dtype=torch.get_default_dtype(), requires_grad=True)
    else:
        x = torch.tensor(x, dtype=torch.get_default_dtype())
        y = torch.tensor(y, dtype=torch.get_default_dtype())
        theta_deg = torch.tensor(angle_deg, dtype=torch.get_default_dtype())

    theta = torch.deg2rad(theta_deg)

    # Position vector
    r_vec = torch.stack([x, y, torch.tensor(0.0, dtype=torch.get_default_dtype())])
    p = r_vec
    p_norm = torch.norm(p)
    p_hat = p / p_norm

    # Dipole moments
    m_E = m_E_mag * torch.stack([torch.sin(theta), torch.cos(theta), torch.tensor(0.0, dtype=torch.get_default_dtype())])
    m_I = m_I_mag * torch.tensor([1.0, 0.0, 0.0], dtype=torch.get_default_dtype())

    # Normalize dipole directions (as in MATLAB)
    m_E_hat = m_E / torch.norm(m_E)
    m_I_hat = m_I / torch.norm(m_I)

    # Interaction matrices
    Z = torch.eye(3) - 5 * torch.outer(p_hat, p_hat)
    D = torch.eye(3) - 3 * torch.outer(p_hat, p_hat)

    # Magnetic interaction scalar
    lambd = (mu0 / (4 * np.pi)) * torch.norm(m_E) * torch.norm(m_I) / (p_norm ** 3)

    # Force (projected along p_hat like in MATLAB)
    term1 = torch.outer(m_E_hat, m_I_hat)
    term2 = torch.outer(m_I_hat, m_E_hat)
    term3 = (m_I_hat @ Z @ m_I_hat) * torch.eye(3)
    F_tensor = (3 * lambd / p_norm) * (term1 + term2 + term3)

    F_m = F_tensor @ p_hat  # this is a vector
    F_m = F_m * p_hat * 0  # ensure it's along the same direction as p

    # Torque
    T_m = torch.cross(lambd * m_I_hat, D @ m_E_hat, dim=0)

    return F_m, T_m, x, y, theta_deg




def pytorch_solver(initial_guesses):
    e_log = []
    best_error = np.inf
    best_x = None
    i =0
    for a in initial_guesses:
        x_var = np.array([0.01, 0.18, a], dtype=float)
        print(f"\nInitial guess: angle = {a}°")
        print(i)
        i+=1
        
        for iter in range(max_iters):


            F_m, T_m, x_t, y_t, theta_t = force_from_paper_torch(x_var[0], x_var[1], x_var[2], requires_grad=True)
            Fx = F_m[1]  # This may be a tensor of size >1
            if Fx.ndim > 0:
                Fx = Fx[1]  # Extract scalar value

            theta_c_hat = dtheta_dF * Fx + dtheta_dT * (-T_m[2])

            error = theta_c_desired - theta_c_hat
            error_scalar = error.item()
            if abs(error_scalar) < tol:
               
                print(f"x = {x_var[0]:.4f} m\ny = {x_var[1]:.4f} m\nangle = {x_var[2]:.2f} deg")
                print(f"Final error = {rad2deg(error_scalar):.4f} deg")
                return x_var[0], x_var[1], x_var[2]

            e_log.append(abs(error_scalar))

            # Compute grads w.r.t x and y via autograd
            grads = torch.autograd.grad(error, [x_t, y_t], retain_graph=True)
            grad_x = grads[0].detach().item()
            grad_y = grads[1].detach().item()

            # Finite-difference angle gradient
            angle_up = x_var[2] + 1.0
            angle_dn = x_var[2] - 1.0

            _, T_up, _, _, _ = force_from_paper_torch(x_var[0], x_var[1], angle_up, requires_grad=False)
            _, T_dn, _, _, _ = force_from_paper_torch(x_var[0], x_var[1], angle_dn, requires_grad=False)

            theta_up = dtheta_dF * 0 + dtheta_dT * (-T_up[2].item())
            theta_dn = dtheta_dF * 0 + dtheta_dT * (-T_dn[2].item())
            grad_angle = (theta_up - theta_dn) / deg2rad(2)

            J_theta = torch.tensor([[grad_x, grad_y, grad_angle]])


            x_t.grad = None
            y_t.grad = None
            theta_t.grad = None


            # Pseudo-inverse of Jacobian
            if torch.linalg.matrix_rank(J_theta) == 1:
                inv_J = J_theta.T / (J_theta @ J_theta.T)
            else:
                inv_J = torch.pinverse(J_theta)

            delta_x = (inv_J @ error.detach().view(1)).flatten()
            delta_x[2] *= 1000  # Angle update scale

            x_var += alpha * delta_x.detach().numpy()

            # Clamp to constraints
            x_var[0] = np.clip(x_var[0], 0, 0.05)
            x_var[1] = np.clip(x_var[1], 0.17, 0.22)
            x_var[2] = np.clip(x_var[2], 0, 180)

            # Track best solution
            if abs(error.item()) < best_error:
                best_error = abs(error.item())
                best_x = x_var.copy()
            if iter % 1000 == 0:
                print("Jacobian:", J_theta)
            # if iter % 1000 == 0:
            #     print(f"Iter {iter}: x = {x_var[0]:.3f}, y = {x_var[1]:.3f}, angle = {x_var[2]:.2f}, "
            #           f"θ̂ = {rad2deg(theta_c_hat.item()):.2f}°, error = {abs(error.item()):.4f}")
            #     print(f"‖J‖ = {torch.norm(J_theta):.4e}, ‖Δx‖ = {torch.norm(delta_x):.4e}")

    print("\nBest solution found:")
    print(f"x = {best_x[0]:.4f} m\ny = {best_x[1]:.4f} m\nangle = {best_x[2]:.2f} deg")
    print(f"Final error = {rad2deg(best_error):.4f} deg")
    return best_x[0],best_x[1], best_x[2]

    # # Plot convergence
    # plt.figure()
    # plt.plot(e_log)
    # plt.xlabel('Iteration')
    # plt.ylabel('Bending Angle Error')
    # plt.title('Convergence of Bending Angle')
    # plt.grid()
    # plt.show()
def beam_theory_checker(x, y, angle, L):
    Fmi, Tmi, _, _, _  = force_from_paper_torch(x, y, angle)
    Fm = Fmi[1]
    Tm = -Tmi[2]
    theta_c_hat = (Fm * L**2) / (2 * Ev * Iv) + (Tm * L) / (Ev * Iv)
    theta = rad2deg(theta_c_hat)
    return theta

if __name__ == '__main__':
    initial_guesses = [45, 0, 180]
    # Constants
    mu0 = 4 * np.pi * 1e-7
    Br = 1.5
    r, h = 0.04, 0.06
    r_i, h_i = 0.0005, 0.005
    V_E = np.pi * r**2 * h
    V_I = np.pi * r_i**2 * h_i
    m_E_mag = (Br * V_E) / mu0
    m_I_mag = (Br * V_I) / mu0

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
    theta_c_desired = deg2rad(24)

    x, y, angle = pytorch_solver(initial_guesses)
    theta = beam_theory_checker(x, y, angle, L_total)
    print(f'Bending angle check: {theta}')
    # F_m, T_m, x, y, theta_deg = force_from_paper_torch(x, y, angle)
    # print(f'Force = {F_m} Torque = {T_m}')
    # x_vals = [0, 0.01, 0.03, 0.05, 0.07]
    # angles = np.linspace(0, 180, 360)

    # for x in x_vals:
    #     torques = []
    #     for angle in angles:
    #         _, T, *_ = force_from_paper_torch(x, 0.25, angle, requires_grad=False)
    #         torques.append(T[2].item())
    #     plt.plot(angles, torques, label=f'x={x:.2f}')

    # plt.xlabel("Rotation Angle (degrees)")
    # plt.ylabel("Torque Z-component")
    # plt.title("Torque vs Angle for Various X-Offsets")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
