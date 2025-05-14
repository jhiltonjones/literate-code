import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from linearise_model import second_order_taylor_state_prediction  
from catheter_system import catheter_update, catheter_params
from compute_desired_trajectory import compute_desired_trajectory

# === MPC Parameters ===
N = 5                           # Horizon steps
Ts = catheter_params['Ts']          # Timestep
magnet_distance = 0.25              # Fixed offset above tip
x0 = np.array([0.03, 0.0, 0.0, 0.0])  # Initial state
total_steps = 500                   # Total MPC steps
step_size = catheter_params['v'] * catheter_params['Ts']


# === Magnet position relative to tip ===
def compute_control_input(x, theta):
    tip = x[:2]
    offset = np.array([0.0, magnet_distance])
    magnet_xy = tip + offset
    return np.hstack((magnet_xy, theta))

# === Cost function ===
def mpc_cost(theta_seq, x0, Xd, N):
    x = x0.copy()
    cost = 0.0

    for t in range(N):
      
        theta = theta_seq[t]
        u_eq = compute_control_input(x, theta)
        
        delta_u = np.zeros(3)
        delta_u[2] = 2.0  
        
        
        x = second_order_taylor_state_prediction(catheter_update, x, u_eq, delta_u, catheter_params)

        tip = x[:2]
        scale = 1000
        error = scale * (tip - Xd[t])
        weight = 1e2 if t == 0 else 10  # more cost on first prediction
        cost += weight * np.linalg.norm(error)**2
        if x[1] > Xd[t][1]:  # tip above desired
            overshoot_penalty = 5 * (x[1] - Xd[t][1])**2
            cost += overshoot_penalty

    return cost

Xd_full, v_suggested = compute_desired_trajectory(k=0, N=total_steps+N, step=step_size)
print("First 5 desired tip_y values:", Xd_full[:5, 1])

x_hist = [x0.copy()]
predicted_tips = []   # stores predicted tip at t+1
desired_tips = []     # stores desired tip at t+1
actual_tips = []      # stores actual tip after applying θ
u_hist = []
theta_guess = np.ones(N) * 90


print("Step | Desired Tip       | Chosen θ (deg) | Predicted Tip")

for step in range(total_steps):
    x_curr = x_hist[-1]

    Xd_horizon = Xd_full[step:step+N, :]

    result = minimize(
        mpc_cost,
        theta_guess,
        args=(x_curr, Xd_horizon, N),
        bounds=[(60, 120)] * N,

        options={'maxiter': 100, 'disp': False}
    )
    theta_opt = result.x

    x_rollout = x_curr.copy()
    theta_t0 = theta_opt[0]
    u_t0 = compute_control_input(x_rollout, theta_t0)
    x_predicted_next = catheter_update(0, x_rollout, u_t0, {**catheter_params, "return_full": False})
    predicted_tips.append(x_predicted_next[:2])
    desired_tips.append(Xd_full[step][:2])          

    print(f"\n--- Step {step} Horizon Rollout ---")
    for t in range(N):
        theta_t = theta_opt[t]
        u_t = compute_control_input(x_rollout, theta_t)
        x_rollout = catheter_update(0, x_rollout, u_t, {**catheter_params, "return_full": False})
        tip_pred = x_rollout[:2]
        tip_desired = Xd_horizon[t]
        
        if t == 0:
            tip_actual = x_curr[:2]
            print(f"  t+{t:2d} | θ: {theta_t:7.2f}° | Actual Tip: ({tip_actual[0]:.4f}, {tip_actual[1]:.4f})")
        print(f"  t+{t:2d} | θ: {theta_t:7.2f}° | Desired: ({tip_desired[0]:.4f}, {tip_desired[1]:.4f}) | Predicted: ({tip_pred[0]:.4f}, {tip_pred[1]:.4f})")

        
    
    theta_opt = result.x
    theta_applied = theta_opt[0]
    u_applied = compute_control_input(x_curr, theta_applied)
    u_hist.append(u_applied)

    x_next = catheter_update(0, x_curr, u_applied, {**catheter_params, "return_full": False})
    actual_tips.append(x_next[:2])   
    if (x_next[0] > Xd_full[-1, 0]) and (x_next[1] > Xd_full[-1, 1]):
        print(f"Stopping at step {step} — tip = ({x_next[0]:.4f}, {x_next[1]:.4f}) exceeded target = ({Xd_full[-1, 0]:.4f}, {Xd_full[-1, 1]:.4f})")
        break


    x_hist.append(x_next)

    tip_pred_y = x_next[1]
    tip_desired_y = Xd_horizon[0][1]
    # print(f"{step:4d} | Tip_y_des: {tip_desired_y:.4f} | θ: {theta_applied:7.2f}° | Tip_y_pred: {tip_pred_y:.4f}")

    theta_guess = np.roll(theta_opt, -1)
    theta_guess[-1] = theta_opt[-1]

# === Plot ===
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
predicted_tips = np.array(predicted_tips)
actual_tips = np.array(actual_tips)
desired_tips = np.array(desired_tips)

plt.figure(figsize=(8, 5))
plt.plot(desired_tips[:, 1], label='Desired Tip Y', linestyle='--')
plt.plot(actual_tips[:, 1], label='Actual Tip Y', marker='o', markersize=3)
plt.plot(predicted_tips[:, 1], label='Predicted Tip Y (1-step)', linestyle=':')
plt.xlabel("MPC Step")
plt.ylabel("Tip y-position (m)")
plt.title("Tip y-position: Actual vs Predicted vs Desired")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
