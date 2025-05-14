import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

# ---------------------------
# Spring-mass system
# ---------------------------
spring_mass_params = {'m': 1.0, 'k': 10.0, 'c': 1.0, 'Ts': 0.1}

def spring_mass_update(t, x, u, params):
    m, k, c, Ts = params['m'], params['k'], params['c'], params['Ts']
    pos, vel = x
    acc = (-k * pos - c * vel + u[0]) / m
    vel_next = vel + acc * Ts
    pos_next = pos + vel_next * Ts
    return np.array([pos_next, vel_next])

def compute_desired_trajectory_spring(k, N, step, Ts=1.0):
    t = np.arange(k, k + N) * Ts
    A, f = 0.5, 0.3
    x = A * np.sin(2 * np.pi * f * t)
    v = 2 * np.pi * f * A * np.cos(2 * np.pi * f * t)
    return x.reshape(-1,1), v.reshape(-1,1)

def get_linear_matrices(f, x0, u0, params, delta=1e-5):
    n, m = len(x0), len(u0)
    A, B = np.zeros((n,n)), np.zeros((n,m))
    f0 = f(0, x0, u0, params)
    for i in range(n):
        dx = np.zeros_like(x0)
        dx[i] = delta
        A[:, i] = (f(0, x0 + dx, u0, params) - f0) / delta
    for j in range(m):
        du = np.zeros_like(u0)
        du[j] = delta
        B[:, j] = (f(0, x0, u0 + du, params) - f0) / delta
    return A, B

# ---------------------------
# Multiple-shooting MPC experiment
# ---------------------------
x0 = np.array([0.0, 0.0])
total_steps = 300
Ts = spring_mass_params['Ts']
positions, velocities = compute_desired_trajectory_spring(0, total_steps, 0, Ts)
Xd_full = np.hstack([positions, velocities])

errors = []
trajectories = {}

for N in [2, 5, 10, 20, 30, 40]:
    x_hist = [x0.copy()]
    u_hist = []
    u_nom = [np.array([0.0])] * N  # initial guess

    for step in range(total_steps):
        if step + N > Xd_full.shape[0]:
            pad = step + N - Xd_full.shape[0]
            Xd_full = np.vstack([Xd_full, np.tile(Xd_full[-1], (pad, 1))])

        x_curr = x_hist[-1]
        Xd_horizon = Xd_full[step:step+N, :]

        for iter in range(10):  # multiple-shooting iterations
            x_nom = [x_curr]
            for i in range(N):
                x_next = spring_mass_update(0, x_nom[-1], u_nom[i], spring_mass_params)
                x_nom.append(x_next)

            A_list, B_list = [], []
            for i in range(N):
                A, B = get_linear_matrices(spring_mass_update, x_nom[i], u_nom[i], spring_mass_params)
                A_list.append(A)
                B_list.append(B)

            n, m = A_list[0].shape[0], B_list[0].shape[1]
            Phi = np.zeros((N*n, n))
            Gamma = np.zeros((N*n, N*m))
            for i in range(N):
                Ai = np.eye(n)
                for j in range(i):
                    Ai = A_list[j] @ Ai
                Phi[i*n:(i+1)*n, :] = Ai
                for j in range(i+1):
                    Aj = np.eye(n)
                    for k in range(j, i):
                        Aj = A_list[k] @ Aj
                    Gamma[i*n:(i+1)*n, j*m:(j+1)*m] = Aj @ B_list[j]

            Xd_flat = Xd_horizon.flatten()
            Q = np.eye(2) * 1e3
            R = np.eye(1) * 1e-1
            P = Q
            Q_stack = np.kron(np.eye(N), Q)
            R_stack = np.kron(np.eye(N), R)

            H = Gamma.T @ Q_stack @ Gamma + R_stack
            f = Gamma.T @ Q_stack @ (Phi @ x_curr - Xd_flat)

            u = cp.Variable(N)
            terminal_state = Phi[-2:, :] @ x_curr + Gamma[-2:, :] @ u
            objective = (0.5 * cp.quad_form(u, H) +
                         f.T @ u +
                         cp.quad_form(terminal_state - Xd_horizon[-1], P))

            prob = cp.Problem(cp.Minimize(objective))
            prob.solve(solver=cp.OSQP)

            if u.value is not None:
                u_nom = [np.array([u_val]) for u_val in u.value]

        # Apply first control after iterations
        u_applied = u_nom[0]
        x_next = spring_mass_update(0, x_curr, u_applied, spring_mass_params)
        x_hist.append(x_next)

    x_hist = np.array(x_hist)
    trajectories[N] = x_hist.copy()

    ref = Xd_full[:x_hist.shape[0], 0]
    sim = x_hist[:, 0]
    rmse = np.sqrt(np.mean((ref - sim)**2))
    errors.append((N, rmse))

# ---------------------------
# Plot error vs horizon
# ---------------------------
Ns, rmse_values = zip(*errors)
plt.figure()
plt.plot(Ns, rmse_values, marker='o')
plt.xlabel('Prediction Horizon N')
plt.ylabel('Tracking RMSE')
plt.title('Tracking Error vs Horizon (multiple-shooting)')
plt.grid()
plt.show()

# ---------------------------
# Plot position trajectories
# ---------------------------
plt.figure(figsize=(10,6))
time = np.arange(total_steps+1) * Ts
ref_positions = Xd_full[:total_steps+1, 0]
plt.plot(time, ref_positions, 'k--', label='Reference trajectory')

for N, traj in trajectories.items():
    plt.plot(time[:len(traj)], traj[:, 0], label=f'N = {N}')

plt.xlabel('Time (s)')
plt.ylabel('Position')
plt.title('Spring-Mass Position Tracking for Different Horizons')
plt.legend()
plt.grid()
plt.show()

# ---------------------------
# Plot velocity trajectories
# ---------------------------
plt.figure(figsize=(10,6))
ref_velocities = Xd_full[:total_steps+1, 1]
plt.plot(time, ref_velocities, 'k--', label='Reference velocity')

for N, traj in trajectories.items():
    plt.plot(time[:len(traj)], traj[:, 1], label=f'N = {N}')

plt.xlabel('Time (s)')
plt.ylabel('Velocity')
plt.title('Spring-Mass Velocity Tracking for Different Horizons')
plt.legend()
plt.grid()
plt.show()
