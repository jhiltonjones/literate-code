import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from catheter_system import catheter_update, catheter_params
from compute_desired_trajectory import compute_desired_trajectory


def compute_drift(x, theta, magnet_distance=0.25):
    magnet_xy = np.array([x[0], x[1] + magnet_distance])
    u = np.concatenate((magnet_xy, [theta]))
    x_next = catheter_update(0, x, u, {**catheter_params, "return_full": False})
    return x_next - x



def plot_tip_vs_theta(x_curr, magnet_xy):
    thetas = np.deg2rad(np.linspace(80, 100, 50))
    y_values = []

    for theta in thetas:
        u = np.concatenate((magnet_xy, [theta]))
        x_next = catheter_update(0, x_curr, u, {**catheter_params, "return_full": False})
        y_values.append(x_next[1])  

    plt.plot(np.rad2deg(thetas), y_values)
    plt.xlabel("θ (degrees)")
    plt.ylabel("Tip y-position")
    plt.title("Tip Height vs Magnet Rotation θ")
    plt.grid(True)
    plt.show()

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


N = 5     
Ts = catheter_params['Ts']       
magnet_distance = 0.25    #reduce sampling time         
x0 = np.array([0.03, 0.0, 0.0, 0.0])
total_steps = 500
step_size = catheter_params['v'] * catheter_params['Ts']
v = catheter_params['v']
Ts = catheter_params['Ts']


Q = np.diag([1e2, 1e2])  

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


def linear_mpc_cost(theta_seq, x0, Xd, A, B):
    cost = 0.0
    x = x0.copy()

    for t in range(N):
        theta = theta_seq[t]
        magnet_xy = np.array([x[0], x[1] + 0.25])  
        u = np.concatenate((magnet_xy, [theta]))
        x = catheter_update(0, x, u, {**catheter_params, "return_full": False})

        tip = x[:2] 
        desired_tip = Xd[t][:2]

        weight = 1e2 if t == 0 else 1e1
        scale = 1000.0
        tip_error = (tip - desired_tip) * scale
        cost += weight * tip_error.T @ Q @ tip_error


        R = 0.1
        cost += R * theta**2

        if t > 0:
            dtheta = theta - theta_seq[t - 1]
            cost += 1e2 * dtheta**2

    terminal_tip = x[:2]
    desired_terminal_tip = Xd[N - 1][:2]
    Q_terminal = np.diag([1e1, 1e1])
    terminal_error = terminal_tip - desired_terminal_tip
    cost += terminal_error.T @ Q_terminal @ terminal_error

    return cost



Xd_full, v_suggested = compute_desired_trajectory(k=0, N=total_steps + N, step=step_size)

x_hist = [x0.copy()]
theta_guess = np.ones(N) * 90
u_hist = []
actual_tips = []

for step in range(total_steps):
    x_curr = x_hist[-1]
    Xd_horizon = Xd_full[step:step+N, :]
    magnet_xy = np.array([x_curr[0], x_curr[1] + magnet_distance])
    u_ref = np.concatenate((magnet_xy, [95])) 
    A, B = compute_AB(x_curr, u_ref)

    res = minimize(
        lambda theta_seq: linear_mpc_cost(theta_seq, x_curr, Xd_horizon, A, B),
        theta_guess,
        bounds=[(30,150)] * N,
        options={'maxiter': 200, 'disp': True}
    )


    theta_opt = res.x
    print(f"\n--- Step {step} Horizon Rollout ---")
    theta_applied = theta_opt[0]
    u_applied = np.concatenate((magnet_xy, [theta_applied]))
    A, B = compute_AB(x_curr, u_applied)
    x_next = catheter_update(0, x_curr, u_applied, {**catheter_params, "return_full": False})
    print(f"\n--- Step {step} Horizon Rollout ---")
    x_pred = x_curr.copy()
    
    predicted_states = predict_trajectory(x_next, theta_opt)

    for t in range(N):
        pred_tip = predicted_states[t][:2]
        desired_tip = Xd_horizon[t][:2]

        if step + t + 1 < len(x_hist):
            actual_tip = x_hist[step + t + 1][:2]
        elif t == 0:
            actual_tip = x_next[:2]
        else:
            actual_tip = None

        if actual_tip is not None:
            actual_str = f" | Actual Tip = ({actual_tip[0]:.4f}, {actual_tip[1]:.4f})"
        else:
            actual_str = ""

        print(f"t = {t:2d} | θ = {theta_opt[t]:.2f} deg | Linear Pred Tip = ({pred_tip[0]:.4f}, {pred_tip[1]:.4f}) | Desired Tip = ({desired_tip[0]:.4f}, {desired_tip[1]:.4f}){actual_str}")

    if (x_next[0] > Xd_full[-1, 0]):
        # print(f"Stopping at step {step} — tip = ({x_next[0]:.4f}, {x_next[1]:.4f}) exceeded target = ({Xd_full[-1, 0]:.4f}, {Xd_full[-1, 1]:.4f})")
        break
    x_hist.append(x_next)
    actual_tips.append(x_next[:2])
    u_hist.append(u_applied)

    theta_guess = np.roll(theta_opt, -1)
    theta_guess[-1] = theta_opt[-1]

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




