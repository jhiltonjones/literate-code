import numpy as np
import matplotlib.pyplot as plt
import control as ct

from catheter_system import catheter_sys, catheter_update, catheter_params
def second_order_taylor_state_prediction(f, x0, u0, delta_u, params, du=1e-4):
    """
    Predict f(x0, u0 + delta_u) using second-order Taylor expansion.
    """
    f0 = f(0, x0, u0, {**params, "return_full": False})
    f0 = np.asarray(f0)
    n_u = len(u0)
    n_x = len(f0)

    # First derivative (Jacobian B)
    B = np.zeros((n_x, n_u))
    for i in range(n_u):
        u_perturbed = u0.copy()
        u_perturbed[i] += du
        f_perturbed = f(0, x0, u_perturbed, {**params, "return_full": False})
        B[:, i] = (np.asarray(f_perturbed) - f0) / du

    # Second derivative (Hessian for each output dim)
    H = np.zeros((n_x, n_u, n_u))
    for i in range(n_u):
        for j in range(n_u):
            u_pp = u0.copy()
            u_pm = u0.copy()
            u_mp = u0.copy()
            u_mm = u0.copy()

            u_pp[i] += du; u_pp[j] += du
            u_pm[i] += du; u_pm[j] -= du
            u_mp[i] -= du; u_mp[j] += du
            u_mm[i] -= du; u_mm[j] -= du

            f_pp = np.asarray(f(0, x0, u_pp, {**params, "return_full": False}))
            f_pm = np.asarray(f(0, x0, u_pm, {**params, "return_full": False}))
            f_mp = np.asarray(f(0, x0, u_mp, {**params, "return_full": False}))
            f_mm = np.asarray(f(0, x0, u_mm, {**params, "return_full": False}))

            H[:, i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * du**2)

    # Apply second-order Taylor expansion
    delta_u = delta_u.reshape(-1, 1)
    first_order = B @ delta_u
    second_order = np.einsum('ijk,j,k->i', H, delta_u[:, 0], delta_u[:, 0])
    return f0 + first_order[:, 0] + 0.5 * second_order


# === Settings ===
magnet_distance = 0.25                    # Fixed vertical distance above tip
perturb_deg = 30           # Small magnet rotation perturbation
steps = 2                          # Number of simulation steps
Ts = catheter_sys.dt                      # Timestep

# === Sweep magnet rotation angles ===
theta_vals = np.linspace(0, 150, 9)
tip_y_linear = []
tip_y_nonlinear = []
bending_angle_nonlinear = []

# Reference catheter initial direction (straight along +x axis)
ref_direction = np.array([1.0, 0.0])  # horizontal rightward

linear_models = []

for theta in theta_vals:
    x0 = np.array([0.03, 0.0, 0.0, 0.0])
    ueq = np.array([x0[0], x0[1] + magnet_distance, theta])
    
    # # === Manual linearization via finite differences ===
    # A, B = second_order_taylor_state_prediction(catheter_update, x0, ueq, catheter_params)
    
    # linear_models.append({
    #     'theta': theta,
    #     'A': A,
    #     'B': B,
    #     'x0': x0,
    #     'u0': ueq
    # })

    # === Linear prediction using Δx = AΔx + BΔu ===
    delta_u = (np.array([0.0, 0.0, perturb_deg])).reshape(-1)
    x_taylor = np.array(x0)  # x stays fixed in model
    for _ in range(steps):
        x_taylor = second_order_taylor_state_prediction(
            catheter_update, x_taylor, ueq, delta_u, catheter_params
        )

    tip_y_taylor = x_taylor[1]
    tip_y_linear.append(tip_y_taylor)

    # === Nonlinear rollout ===
    u_perturbed = ueq.copy()
    u_perturbed[2] += perturb_deg
    x_nl = x0.copy()
    for _ in range(steps):
        x_nl, gamma_bending, theta_c_hat = catheter_update(0, x_nl, u_perturbed, {**catheter_params, "return_full": True})

    tip_y_nonlinear.append(x_nl[1])
    bending_angle_nonlinear.append(np.rad2deg(gamma_bending))
    errors = np.array(tip_y_nonlinear) - np.array(tip_y_linear)


# plt.figure()
# plt.plot(theta_vals, tip_y_linear, 'o-', label='Linearized Prediction (tip_y)')
# plt.plot(theta_vals, tip_y_nonlinear, 'x--', label='Nonlinear Simulation (tip_y)')
# plt.axhline(0, color='black', linestyle='--')
# plt.xlabel('Magnet Rotation (degrees)')
# plt.ylabel('Tip_y (m)')
# plt.title('Linear vs Nonlinear Tip_y Prediction')
# plt.grid(True)
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(theta_vals, errors, 'ro-')
# plt.xlabel("Magnet rotation angle (θ)")
# plt.ylabel("tip_y error (nonlinear - linear)")
# plt.title("Linearization Error vs θ")
# plt.grid(True)
# plt.show()
# # === Plot Bending Angle vs Magnet Rotation ===
# plt.figure(figsize=(10,6))
# plt.plot(theta_vals, bending_angle_nonlinear, 's-', label='Bending Angle (from initial)')
# plt.axhline(0, color='black', linestyle='--')
# plt.xlabel('Magnet Command Rotation (degrees)')
# plt.ylabel('Catheter Bending Angle (degrees)')
# plt.title('Catheter Bending Angle vs Magnet Command Rotation')
# plt.grid(True)
# plt.legend()
# plt.show()
# === Visualize Catheter Bending Geometry for Each Magnet Rotation ===
# plt.figure(figsize=(15, 15))
# for idx, theta in enumerate(theta_vals):
#     # Reset catheter to straight initial position
#     x0 = np.array([0.03, 0.0, 0.0, 0.0])
#     ueq = np.array([x0[0], x0[1] + magnet_distance, theta])
#     x_vis = x0.copy()

#     for _ in range(steps):
#         x_vis, gamma_vis, theta_vis = catheter_update(0, x_vis, ueq, {**catheter_params, "return_full": True})

#     tip = x_vis[0:2]
#     base = x_vis[2:4]
#     magnet = np.array([x0[0], x0[1] + magnet_distance])

#     plt.subplot(4, 3, idx + 1)
#     plt.plot([base[0], tip[0]], [base[1], tip[1]], 'b-', label='Catheter')
#     plt.plot(tip[0], tip[1], 'bo', label='Tip')
#     plt.plot(base[0], base[1], 'go', label='Base')
#     plt.plot(magnet[0], magnet[1], 'ro', label='Magnet')

#     # Optional: draw magnetic force direction (simplified as downward arrow)
#     plt.arrow(magnet[0], magnet[1], 0, -0.02, head_width=0.002, head_length=0.01, color='r')

#     plt.title(f"Magnet θ = {theta:.0f}° | Tip_y = {tip[1]:.3f}m")
#     plt.axis('equal')
#     plt.grid(True)
#     plt.xlim([0.0, 0.06])
#     plt.ylim([-0.02, 0.25])
#     if idx == 0:
#         plt.legend()

# plt.suptitle("Catheter and Magnet Geometry for Different Rotation Angles", fontsize=16)
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()
# threshold = 1e-3  # 1 mm
# steps = 2
# magnet_distance = 0.2
# theta_vals = np.linspace(0, 130, 10)
# max_valid_perturb = []

# for theta in theta_vals:
#     x0 = np.array([0.03, 0.0, 0.0, 0.0])
#     ueq = np.array([x0[0], x0[1] + magnet_distance, theta])
#     found = False

#     for perturb_deg in range(1, 91):  # Perturb from 1 to 90 degrees
#         delta_u = np.array([0.0, 0.0, perturb_deg])
#         x_taylor = np.array(x0)

#         for _ in range(steps):
#             x_taylor = second_order_taylor_state_prediction(
#                 catheter_update, x_taylor, ueq, delta_u, catheter_params
#             )

#         tip_y_taylor = x_taylor[1]

#         # Nonlinear rollout
#         u_perturbed = ueq.copy()
#         u_perturbed[2] += perturb_deg
#         x_nl = np.array(x0)
#         for _ in range(steps):
#             x_nl, gamma_bending, theta_c_hat = catheter_update(
#                 0, x_nl, u_perturbed, {**catheter_params, "return_full": True}
#             )

#         tip_y_nonlinear = x_nl[1]
#         error = np.abs(tip_y_nonlinear - tip_y_taylor)

#         if error > threshold:
#             max_valid_perturb.append(perturb_deg - 1)
#             found = True
#             break

#     if not found:
#         max_valid_perturb.append(90)

# # Plot results
# plt.figure(figsize=(10,6))
# plt.plot(theta_vals, max_valid_perturb, 'bo-')
# plt.xlabel("Base Magnet Angle θ (deg)")
# plt.ylabel("Max Perturbation Before >1mm Error (deg)")
# plt.title("How Far You Can Perturb Before Nonlinear Deviation > 1mm")
# plt.grid(True)
# plt.show()
