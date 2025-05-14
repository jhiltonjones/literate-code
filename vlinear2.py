import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from catheter_system import catheter_update, catheter_params
from compute_desired_trajectory import compute_desired_trajectory
import cvxpy as cp




def build_cost_matrices(Phi, Gamma, x0, d_vec, Xd, Q, R, S=None, N_dyn=None):
    if N_dyn is None:
        N_dyn = Xd.shape[0]

    Xd_stack = Xd.flatten()
    Q_full = np.zeros((4, 4))      
    scale = 10000.0
    Q_full[:2, :2] = Q * scale**6

    Q_stack = np.kron(np.eye(N_dyn), Q_full)
    H = Gamma.T @ Q_stack @ Gamma + np.kron(np.eye(N_dyn), R)

    f = Gamma.T @ Q_stack @ (Phi @ x0 + d_vec - Xd_stack)

    if S is not None:
        D = np.eye(N_dyn) - np.roll(np.eye(N_dyn), 1, axis=1)
        D[0] = 0  
        H += D.T @ np.kron(np.eye(N_dyn), S) @ D

    return H, f, d_vec


def build_drift_vector_ltv(A_seq, B_seq, x0, u0_seq_full, f):
    N = len(A_seq)
    n = x0.shape[0]
    d_stack = np.zeros((N * n,))
    x_sim = x0.copy()

    for i in range(N):
        u_full = u0_seq_full[i]    
        theta_only = u_full[-1:]                 
        A_i = A_seq[i]
        B_i = B_seq[i]                      

        fx = f(0, x_sim, u_full, {**catheter_params, "return_full": False})
        dx = fx - A_i @ x_sim - B_i @ theta_only
        x_sim = fx.copy()

        for j in range(i + 1):
            A_prod = np.eye(n)
            for k in range(j + 1, i):
                A_prod = A_seq[k] @ A_prod
            d_stack[i * n:(i + 1) * n] += A_prod @ dx
    return d_stack


def build_prediction_matrices_ltv(A_seq, B_seq, N):

    n = A_seq[0].shape[0] 
    m = B_seq[0].shape[1]  

    Phi = np.zeros((N * n, n))
    Gamma = np.zeros((N * n, N * m))

    for i in range(N):
        
        Ai_prod = np.eye(n)
        for j in range(i):
            Ai_prod = A_seq[j] @ Ai_prod
        Phi[i*n:(i+1)*n, :] = Ai_prod

        for j in range(i + 1):
         
            A_prod = np.eye(n)
            for k in range(j+1, i):
                A_prod = A_seq[k] @ A_prod
            Gamma[i*n:(i+1)*n, j*m:(j+1)*m] = A_prod @ B_seq[j]

    return Phi, Gamma


def get_linear_matrices(f, x0, u0, params, delta=25):
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
magnet_distance = 0.25          
x0 = np.array([0.03, 0.0, 0.0, 0.0])
total_steps = 300
step_size = catheter_params['v'] * catheter_params['Ts']
v = catheter_params['v']
Ts = catheter_params['Ts']
R = np.array([[1e2]])       
S = np.array([[1e2]])      
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
dy_tip_vals = []
dy_true_list = []
dy_pred_list = []
sign_match_list = []
nonlinear_dy_pred_list = []
for step in range(total_steps):

    N_dyn = N

    for lookahead in range(1, N + 1):
        horizon_end = step + lookahead
        if horizon_end >= Xd_full.shape[0]:
            N_dyn = lookahead
            break
        last_desired = Xd_full[horizon_end - 1]
        final_target = Xd_full[-1]
        dist_to_final = np.linalg.norm(last_desired[:2] - final_target[:2])
        if dist_to_final < 1e-3:  
            N_dyn = lookahead
            break

    x_curr = x_hist[-1]
    Xd_horizon = Xd_full[step:step+N_dyn, :]
    magnet_xy = np.array([x_curr[0], x_curr[1] + magnet_distance])
    # u_ref = np.concatenate((magnet_xy, [85]))

    A_seq = []
    B_seq = []
    x_temp = x_curr.copy()
    for t in range(N_dyn):
        magnet_xy = np.array([x_temp[0], x_temp[1] + magnet_distance])
        # u_t = np.concatenate((magnet_xy, [85]))  # nominal θ for linearization

        if step == 0 and t == 0:
            theta_linearize = 85  
        elif t == 0:
            theta_linearize = theta_applied 
        else:
            theta_linearize = theta_guess[t - 1]  

        u_t = np.concatenate((magnet_xy, [theta_linearize]))

        A_t, B_t = compute_AB(x_temp, u_t)
        B_t = B_t[:, -1:] 


        A_seq.append(A_t)
        B_seq.append(B_t)

        x_temp = catheter_update(0, x_temp, u_t, {**catheter_params, "return_full": False})


    Phi, Gamma = build_prediction_matrices_ltv(A_seq, B_seq, N_dyn)

    Xd_horizon_full = np.zeros((N_dyn, 4))
    Xd_horizon_full[:, :2] = Xd_horizon

    u0_seq = [np.concatenate((magnet_xy, [85.0]))] * N_dyn  

    d_vec = build_drift_vector_ltv(A_seq, B_seq, x_curr, u0_seq, catheter_update)

    H, f, _ = build_cost_matrices(Phi, Gamma, x_curr, d_vec, Xd_horizon_full, Q, R, S, N_dyn=N_dyn)


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
    B = B[:, -1:]
    x_sim_nl = catheter_update(0, x_curr, np.concatenate((magnet_xy, [theta_opt[0]])), {**catheter_params, "return_full": False})
    dy_pred_nl = x_sim_nl[1] - x_curr[1]
    nonlinear_dy_pred_list.append(dy_pred_nl)
    x_next = catheter_update(0, x_curr, u_applied, {**catheter_params, "return_full": False})
    x_pred = x_curr.copy()
    
    predicted_states = predict_trajectory(x_next, theta_opt)

    pred_stack = (Phi @ x_curr + d_vec + Gamma @ theta_opt).reshape(N_dyn, -1)

    dy_true = x_next[1] - x_curr[1]
    dy_pred = (Phi @ x_curr + d_vec + Gamma @ theta_opt)[1] - x_curr[1]

    dy_true_list.append(dy_true)
    dy_pred_list.append(dy_pred)
    sign_match_list.append(np.sign(dy_true) == np.sign(dy_pred))


    for t in range(N_dyn):
        pred_tip = pred_stack[t][:2] 
        if t == 0:
            dy_true = x_next[1] - x_curr[1]
            dy_pred = (Phi @ x_curr + d_vec + Gamma @ theta_opt)[1] - x_curr[1]
            direction_match = np.sign(dy_pred) == np.sign(dy_true)

            # print(f"Pred Δy = {dy_pred:.4e}, True Δy = {dy_true:.4e}, Match: {direction_match}")
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
    
    dy_tip_vals.append(B[1, -1])

x_hist = np.array(x_hist)
Xd_eval = Xd_full[:len(x_hist), :2] 
errors = np.linalg.norm(x_hist[:, :2] - Xd_eval, axis=1)

average_error = np.mean(errors)
print(f"\n✅ Average Tip Tracking Error = {average_error:.6f} meters")


x_hist = np.array(x_hist)

theta_inputs = np.array([u[-1] for u in u_hist])

plt.figure()
plt.plot(dy_tip_vals)
# plt.plot(theta_inputs, label="θ (deg)", alpha=0.5)
plt.xlabel('Time step')
plt.ylabel('∂y_tip / ∂θ')
plt.title('Sensitivity of Tip Y-position to Rotation Angle')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 4))
time_vector = np.arange(len(theta_inputs)) * Ts
plt.plot(time_vector, theta_inputs, label='Magnet Rotation θ (degrees)')
plt.xlabel('Time (s)')
plt.ylabel('Rotation Angle (°)')
plt.title('Magnet Rotation Input Over Time')
plt.grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure()
plt.plot(errors, label='Tip Error per Step')
plt.xlabel('Time Step')
plt.ylabel('Error (meters)')
plt.title('Tracking Error Over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

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

thetas = np.linspace(30, 150, 50)
y_tips = []

for theta in thetas:
    u = np.concatenate((magnet_xy, [theta]))
    x_next = catheter_update(0, x_curr, u, {**catheter_params, "return_full": False})
    y_tips.append(x_next[1])
dy_dtheta = np.gradient(y_tips, thetas)



plt.figure(figsize=(8, 4))
plt.plot(thetas, y_tips, label='y_tip(θ)')
plt.xlabel('θ (deg)')
plt.ylabel('y_tip')
plt.title('Effect of θ on Tip Y Position (nonlinear)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(thetas, dy_dtheta, color='orange', label='∂y_tip / ∂θ')
plt.xlabel('θ (deg)')
plt.ylabel('dy_tip / dθ')
plt.title('Numerical Derivative of Tip Y Position w.r.t θ')
plt.grid(True)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
plt.legend()
plt.tight_layout()
plt.show()



plt.figure(figsize=(8, 4))
plt.plot(dy_true_list, label="True Δy", color='orange')
plt.plot(dy_pred_list, label="Predicted Δy", color='red', linestyle='--')
plt.xlabel('Episode Step')
plt.ylabel('Δy_tip (m)')
plt.title('True vs Predicted Δy_tip (First Step of Each Horizon)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 2))
plt.bar(range(len(sign_match_list)),
        [1 if match else -1 for match in sign_match_list],
        color=['green' if match else 'red' for match in sign_match_list])
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('Episode Step')
plt.ylabel('Sign Match')
plt.title('Sign Match Between Predicted and True Δy_tip')
plt.tight_layout()
plt.grid(True, axis='y')
plt.show()

# plt.figure(figsize=(10, 5))
# steps = np.arange(len(dy_true_list))

# plt.plot(steps, dy_true_list, label='True Δy', color='orange')
# plt.plot(steps, dy_pred_list, label='Linear Predicted Δy', linestyle='--', color='red')
# plt.plot(steps, nonlinear_dy_pred_list, label='Nonlinear Predicted Δy', linestyle='-.', color='blue')

# plt.xlabel('Episode Step')
# plt.ylabel('Δy_tip (m)')
# plt.title('True vs Predicted Δy_tip (First Step of Each Horizon)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
