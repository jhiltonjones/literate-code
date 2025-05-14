import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from catheter_system import catheter_update, catheter_params
from compute_desired_trajectory import compute_desired_trajectory
import cvxpy as cp

def build_drift_vector(A, B, x0, u0, N, f):
    d = f(0, x0, u0, {**catheter_params, "return_full": False}) - A @ x0 - B @ u0
    n = x0.shape[0]
    d_stack = np.zeros((N * n,))
    for i in range(N):
        for j in range(i + 1):
            d_stack[i * n:(i + 1) * n] += np.linalg.matrix_power(A, i - j) @ d
    return d_stack


def build_cost_matrices(Phi, Gamma, A, B, x0, u0, Xd, Q, R, S=None, N_dyn=None):
    if N_dyn is None:
        N_dyn = Xd.shape[0]

    Xd_stack = Xd.flatten()
    Q_full = np.zeros((4, 4))      # full state size
    scale = 10000.0
    Q_full[:2, :2] = Q * scale**2  # correct scaling of tip error
           # apply tip cost
    Q_stack = np.kron(np.eye(N_dyn), Q_full)
    H = Gamma.T @ Q_stack @ Gamma + np.kron(np.eye(N_dyn), R)
    d_vec = build_drift_vector(A, B, x0, u0, N_dyn, catheter_update)
    f = Gamma.T @ Q_stack @ (Phi @ x0 + d_vec - Xd_stack)
    if S is not None:
        D = np.eye(N_dyn) - np.roll(np.eye(N_dyn), 1, axis=1)
        D[0] = 0  # no penalty on first θ
        H += D.T @ np.kron(np.eye(N_dyn), S) @ D
    return H, f, d_vec


def build_prediction_matrices(A, B, N):
    n = A.shape[0]
    m = 1  # only optimizing theta
    Phi = np.zeros((N * n, n))
    Gamma = np.zeros((N * n, N * m))

    for i in range(N):
        Phi[i*n:(i+1)*n, :] = np.linalg.matrix_power(A, i+1)
        for j in range(i+1):
            Gamma[i*n:(i+1)*n, j*m:(j+1)*m] = np.linalg.matrix_power(A, i - j) @ B[:, -1:]  # Only last col (theta)
    return Phi, Gamma

def get_linear_matrices(f, x0, u0, params, delta=1e-5):
    n = len(x0)
    m = len(u0)
    A = np.zeros((n, n))
    B = np.zeros((n, m))

    f0 = f(0, x0, u0, {**params, "return_full": False})

    for i in range(n):
        dx = np.zeros_like(x0)
        dx[i] = delta
        A[:, i] = (f(0, x0 + dx, u0, {**params, "return_full": False}) - f0) / delta

    for j in range(m):
        du = np.zeros_like(u0)
        du[j] = delta
        B[:, j] = (f(0, x0, u0 + du, {**params, "return_full": False}) - f0) / delta

    return A, B


N = 10
Ts = catheter_params['Ts']       
magnet_distance = 0.25    #reduce sampling time         
x0 = np.array([0.03, 0.0, 0.0, 0.0])
total_steps = 400
step_size = catheter_params['v'] * catheter_params['Ts']
v = catheter_params['v']
Ts = catheter_params['Ts']
R = np.array([[0.1]])       # Regularization on θ
S = np.array([[1e1]])       # Smoothing term (optional)
Q = np.diag([1e3, 1e3])  

def compute_AB(x_ref, u_ref):
    A, B = get_linear_matrices(catheter_update, x_ref, u_ref, catheter_params)
    return A, B

def predict_trajectory(x0, theta_seq):
    X = [x0]
    for theta in theta_seq:
        x_curr = X[-1]
        magnet_xy = np.array([x_curr[0], x_curr[1] + 0.25])
        u = np.concatenate((magnet_xy, [theta]))
        x_next = catheter_update(0, x_curr, u, {**catheter_params, "return_full": False})
        X.append(x_next)
    return np.array(X[1:])

Xd_full, v_suggested = compute_desired_trajectory(k=0, N=total_steps + N, step=step_size)

x_hist = [x0.copy()]
theta_guess = np.ones(N) * 90
u_hist = []
actual_tips = []

for step in range(total_steps):

    # Start with full horizon
    N_dyn = N

    # Look ahead to see if the final point in horizon is the terminal target
    for lookahead in range(1, N + 1):
        horizon_end = step + lookahead
        if horizon_end >= Xd_full.shape[0]:
            N_dyn = lookahead
            break
        last_desired = Xd_full[horizon_end - 1]
        final_target = Xd_full[-1]
        dist_to_final = np.linalg.norm(last_desired[:2] - final_target[:2])
        if dist_to_final < 1e-4:  # within 0.1 mm or 100 microns
            N_dyn = lookahead
            break

    x_curr = x_hist[-1]
    Xd_horizon = Xd_full[step:step+N_dyn, :]
    magnet_xy = np.array([x_curr[0], x_curr[1] + magnet_distance])
    u_ref = np.concatenate((magnet_xy, [85]))

    A, B = compute_AB(x_curr, u_ref)


    Phi, Gamma = build_prediction_matrices(A, B, N_dyn)
    Xd_horizon_full = np.zeros((N_dyn, 4))
    Xd_horizon_full[:, :2] = Xd_horizon

    H, f, d_vec = build_cost_matrices(Phi, Gamma, A, B, x_curr, u_ref, Xd_horizon_full, Q, R, S, N_dyn=N_dyn)


    theta = cp.Variable(N_dyn)
    objective = 0.5 * cp.quad_form(theta, H) + f @ theta
    constraints = [theta >= 30, theta <= 150]
    theta.value = theta_guess[:N_dyn]
    prob = cp.Problem(cp.Minimize(objective), constraints)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        prob.solve(solver=cp.OSQP, verbose=False)

    theta_opt = theta.value

    theta_applied = theta_opt[0]
    u_applied = np.concatenate((magnet_xy, [theta_applied]))
    A, B = compute_AB(x_curr, u_applied)
    

    x_next = catheter_update(0, x_curr, u_applied, {**catheter_params, "return_full": False})
    x_pred = x_curr.copy()
    
    predicted_states = predict_trajectory(x_next, theta_opt)

    # Use the optimizer's internal model: Phi x + d + Gamma theta
    pred_stack = (Phi @ x_curr + d_vec + Gamma @ theta_opt).reshape(N_dyn, -1)

    for t in range(N_dyn):
        pred_tip = pred_stack[t][:2]  # linear prediction with drift
        desired_tip = Xd_horizon[t][:2]

        if step + t + 1 < len(x_hist):
            actual_tip = x_hist[step + t + 1][:2]
        elif t == 0:
            actual_tip = x_next[:2]
        else:
            actual_tip = None

        actual_str = f" | Actual Tip = ({actual_tip[0]:.4f}, {actual_tip[1]:.4f})" if actual_tip is not None else ""

        if step % 100 == 0:
            print(f"\n--- Step {step} Horizon Rollout ---")
            print(f"t = {t:2d} | θ = {theta_opt[t]:.2f} deg | Linear Pred Tip = ({pred_tip[0]:.4f}, {pred_tip[1]:.4f}) | Desired Tip = ({desired_tip[0]:.4f}, {desired_tip[1]:.4f}){actual_str}")
            print(f"d(y_tip)/dθ ≈ {B[1, -1]:.4e}")
    if (x_next[0] > Xd_full[-1, 0]):
        # print(f"Stopping at step {step} — tip = ({x_next[0]:.4f}, {x_next[1]:.4f}) exceeded target = ({Xd_full[-1, 0]:.4f}, {Xd_full[-1, 1]:.4f})")
        break
    x_hist.append(x_next)
    actual_tips.append(x_next[:2])
    u_hist.append(u_applied)

    theta_guess = np.roll(theta_opt, -1)
    theta_guess = np.append(theta_guess[:-1], theta_guess[-1])

x_hist = np.array(x_hist)
Xd_eval = Xd_full[:len(x_hist), :2]  

errors = np.linalg.norm(x_hist[:, :2] - Xd_eval, axis=1)

average_error = np.mean(errors)
print(f"\n✅ Average Tip Tracking Error = {average_error:.6f} meters")


x_hist = np.array(x_hist)
plt.figure()
plt.plot(errors, label='Tip Error per Step')
plt.xlabel('Time Step')
plt.ylabel('Error (meters)')
plt.title('Tracking Error Over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

x_hist = np.array(x_hist)
plt.figure(figsize=(6, 4))
plt.plot(x_hist[:, 0], x_hist[:, 1], 'r-o', label='MPC Tip Path')
plt.plot(Xd_full[:total_steps, 0], Xd_full[:total_steps, 1], 'k--', label='Desired')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.title('MPC Trajectory Tracking (Tip)')
plt.tight_layout()
plt.show()