import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from catheter_system import catheter_update, catheter_params

magnet_distance = 0.25  
def compute_control_input(x, theta):
    tip = x[:2]
    offset = np.array([0.0, magnet_distance])
    magnet_xy = tip + offset
    return np.hstack((magnet_xy, theta))

def get_linear_matrices(f, x0, u0, params, delta=1e-5):
    n = len(x0)
    m = len(u0)
    A = np.zeros((n, n))
    B = np.zeros((n, m))
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

# Placeholder params
catheter_params["return_full"] = False  # Merge flag

# Base state and magnet rotation
x0 = np.array([0.03, 0.0, 0.0, 0.0])
theta_fixed = np.deg2rad(80)
x_range = np.linspace(-0.05, 0.05, 20)
y_range = np.linspace(-0.05, 0.05, 20)

# Grid for position perturbation
X, Y = np.meshgrid(x_range, y_range)
Z_error = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        magnet_xy = np.array([x0[0] + X[i, j], x0[1] + Y[i, j]])
        u = compute_control_input(x0, theta_fixed)


        x_nl = catheter_update(0, x0, u, {**catheter_params, "return_full": False})

        A, B = get_linear_matrices(catheter_update, x0, u, catheter_params)
        x_lin = A @ x0 + B @ u

        # Tip error (only x and y)
        error = np.linalg.norm(x_nl[:2] - x_lin[:2])
        Z_error[i, j] = error

# Plotting
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z_error, cmap='viridis')
ax.set_xlabel('Δx (m)')
ax.set_ylabel('Δy (m)')
ax.set_zlabel('Tip Position Error (m)')
ax.set_title('Linear Prediction Error vs Magnet Position Perturbation')
plt.tight_layout()
plt.show()
