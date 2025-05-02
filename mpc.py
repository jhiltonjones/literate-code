import control as ct
import control.optimal as opt
import numpy as np
import matplotlib.pyplot as plt
from catheter_system import catheter_update, catheter_params
from compute_desired_trajectory import compute_desired_trajectory
import control
print(control.__version__)

# === MPC Horizon & Trajectory ===
N = 10
step = 0.005
Ts = catheter_params["Ts"]
Xd, v_suggested = compute_desired_trajectory(k=0, N=N, step=step)

# === Initial State ===
x0 = np.array([0.03, 0.0, 0.0, 0.0])  # tip_x, tip_y, base_x, base_y
magnet_x = x0[0]
magnet_y = x0[1] + 0.25
def magnet_input_constraints(t, x, u):
    theta = u[2]
    theta_lb = np.deg2rad(80)
    theta_ub = np.deg2rad(100)
    return [theta - theta_lb], [theta_ub - theta]

def make_tracking_cost(Xd):
    def tracking_cost(x, u, t=0, params=None):
        idx = min(int(t), Xd.shape[0] - 1)
        x_des = Xd[idx]
        tip_error = x[:2] - x_des
        # Small penalty on magnet position and angle change
        u_penalty = 0.01 * np.sum(u[:2]**2) + 0.001 * u[2]**2
        return float(np.sum(tip_error**2) + u_penalty)
    return tracking_cost


tracking_cost_fn = make_tracking_cost(Xd)

# === Constraint: limit rotation angle ===
def magnet_rotation_constraint(t, x, u):
    theta = u[0]
    lb = np.deg2rad(80)
    ub = np.deg2rad(100)
    return [theta - lb], [ub - theta]

input_constraints = [magnet_rotation_constraint]

# === WRAP SYSTEM with fixed magnet x/y ===
def catheter_theta_only_update(t, x, u, params):
    mx, my = params["magnet_x"], params["magnet_y"]
    u_full = np.array([mx, my, u[0]])
    return catheter_update(t, x, u_full, catheter_params)

def catheter_theta_only_output(t, x, u, params):
    return x

catheter_sys_theta_only = ct.nlsys(
    catheter_theta_only_update,
    catheter_theta_only_output,
    inputs=3,
    states=4,
    outputs=4,
    dt=catheter_params["Ts"],
    name="catheter_theta_only",
    params={  # ✅ Add this
        "magnet_x": magnet_x,
        "magnet_y": magnet_y,
        "Xd": Xd
    }
)
res = opt.solve_optimal_trajectory(
    catheter_sys_theta_only,
    timepts=np.arange(N),
    x0=x0,
    cost=tracking_cost_fn,
    constraints=[magnet_input_constraints],  # Add others if needed
)



# === Plot Result ===
if res.success:
    print("MPC optimization successful.")
    X_opt = res.states
    U_opt = res.inputs
else:
    raise RuntimeError(f"MPC failed: {res.message}")

plt.figure()
plt.plot(Xd[:, 0], Xd[:, 1], 'b--', label="Desired Trajectory")
plt.plot(X_opt[:, 0], X_opt[:, 1], 'ro-', label="Optimized Tip Trajectory")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Catheter MPC Tracking Result")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.show()
plt.figure()
