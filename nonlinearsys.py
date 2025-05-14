import control as ct
import numpy as np
import matplotlib.pyplot as plt
from catheter_system import catheter_sys, catheter_update, catheter_params

print(catheter_sys)
x0 = np.array([0.03, 0.0, 0.0, 0.0])  # Initial [tip_x, tip_y, base_x, base_y]
u0 = np.array([0.03, 0.0, 0.0])        # Magnet at (0.05, 0.0), angle 0 deg

x1 = catheter_update(0, x0, u0, catheter_params)
print("Next state:", x1)


# Time settings
T_total = 350  # total simulation time (seconds)
Ts = catheter_sys.dt  # timestep
steps = int(T_total / Ts)

# Create time vector
time = np.arange(0, T_total, Ts)

# === Define input trajectory (magnet path) ===
# For example: move magnet slowly to the right
u_trajectory = np.zeros((steps, 3))  # [magnet_x, magnet_y, magnet_angle]
# Fixed offset distance from tip to magnet (in meters)
# === Simulate with magnet always fixed distance from tip ===
magnet_distance = 0.3# 20 cm above tip
magnet_hist = np.zeros((steps, 2))  # record [magnet_x, magnet_y] over time
x_hist = np.zeros((steps+1, len(x0)))
x_hist[0, :] = x0

for k in range(steps):
    tip = x_hist[k, 0:2]
    
    # Move magnet relative to current tip
    magnet_x = tip[0]
    magnet_y = tip[1] + magnet_distance
    if k < steps/3:
        magnet_angle = 85 # facing downward
    elif k<2*steps/3:
        magnet_angle = 100
    else:
        magnet_angle = 75

    u = np.array([magnet_x, magnet_y, magnet_angle])

    x_next = catheter_update(0, x_hist[k], u, catheter_params)
    x_hist[k+1] = x_next
    magnet_hist[k, :] = np.array([magnet_x, magnet_y])

    print(f"Step {k}, Tip: {tip}, Magnet: {[magnet_x, magnet_y]}, Next: {x_next}")


# === Extract tip and base positions ===
tip_hist = x_hist[:, 0:2]   # tip_x, tip_y
base_hist = x_hist[:, 2:4]  # base_x, base_y

# === Plot the catheter motion ===
plt.figure(figsize=(8, 6))
plt.plot(base_hist[:,0], base_hist[:,1], label='Base Path', marker='o')
plt.plot(tip_hist[:,0], tip_hist[:,1], label='Tip Path', marker='x')
plt.plot(magnet_hist[:,0], magnet_hist[:,1], '--', label='Magnet Path')

plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Catheter Tip and Base Motion Over Time')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
