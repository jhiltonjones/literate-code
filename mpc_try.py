import numpy as np
import cvxpy as cp
import time
import matplotlib.pyplot as plt
from bfgs_minimise import compute_angle  # Import correct angle calculation
from ik_modelling import compute_unit_position_vector, components, moment_cath1, compute_center_of_catheter, compute_T_m
from constants import *
from kinematic import bending_moment_equation, compute_curvature, function_alpha_o, force_12
from magnetic import compute_torque, magnetic_moment, volume_calculator_cyclinder, magnetic_field_external_magnet

def mpc(target, actual, EI, theta_l, length_c, x_p, y_p, x_basis, y_basis, p_norm1, time_elapsed):

    N = 10 
    A = 0.9 
    B = 0.05 
    lambda_reg = 0.1 
    u_max = 10  

    length_c += 0.01 

    theta_catheter = cp.Variable(N+1) 
    alpha_star_opt = cp.Variable(N)  

    theta_init = actual

    cost = 0
    constraints = [theta_catheter[0] == cp.Constant(theta_init)] 

    for i in range(N):
        constraints += [theta_catheter[i+1] == A * theta_catheter[i] + B * alpha_star_opt[i]]
        
        cost += cp.square(theta_catheter[i] - target) + lambda_reg * cp.square(alpha_star_opt[i])

        constraints += [cp.abs(alpha_star_opt[i]) <= u_max]

    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.ECOS)

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        print("⚠️ Warning: MPC solver failed, returning 0 rotation.")
        return 0, theta_init, length_c  

    optimal_magnet_adjustment = alpha_star_opt.value[0]
    kappa = compute_curvature(theta_l, length_c)
    print(f"Curvature kappa: {kappa}")
    volume_cath = volume_calculator_cyclinder((s_c/2), length_c_m)
    volume_mag = volume_calculator_cyclinder((h_a/2), d_a)
    print("Volume of magnet is: ", volume_mag)
    print("Volume of catheter is: ", volume_cath)


    x_c, y_c = compute_center_of_catheter(length_c_m, kappa, theta_l)
    print("Center of catheter: ", x_c, y_c)
    # p_hat = compute_unit_position_vector(x_c, y_c, d, h)
    # x_p, y_p, _ = p_hat  
    print(f"x_c: {x_c}, y_c: {y_c}")
    
    print(f"d: {d}, h: {h}")

    p_hat, p_norm1, y_p, x_p = compute_unit_position_vector(x_c, y_c, d, h)
    print(f"x_p: {x_p}, y_p: {y_p}")

    p_unit = x_p * x_basis + y_p * y_basis
    new_magnet_angle = compute_angle(EI, theta_l + optimal_magnet_adjustment, length_c, x_p, y_p, x_basis, y_basis, p_norm1)

    return new_magnet_angle, theta_catheter.value[1], length_c  

time_steps = 20  
time_points = np.linspace(0, time_steps - 1, num=time_steps) 

target_angles = np.linspace(10, 45, num=time_steps)  
catheter_angles = []  
catheter_lengths = []  
magnet_rotations = []  

length_c = 0.05 
x_p, y_p = 0.1, 0.1  
x_basis, y_basis = np.array([1, 0]), np.array([0, 1])  
p_norm1 = 0.2  
previous_magnet_angle = 0 

catheter_angle = target_angles[0] + np.random.uniform(-5, 5)

for t in range(time_steps):
    print(f"\n⏳ Time Step: {t} sec")
    
    d += 0.005

    target_angle = target_angles[t]

    noisy_actual_angle = target_angle + np.random.uniform(-5, 5)

    new_magnet_angle, catheter_angle, length_c = mpc(
        target_angle, noisy_actual_angle, 1.0, np.radians(catheter_angle),
        length_c, x_p, y_p, x_basis, y_basis, p_norm1, t
    )

    rotation_adjustment = new_magnet_angle - previous_magnet_angle
    previous_magnet_angle = new_magnet_angle 

    print(f"Target Angle: {target_angle:.2f} degrees")
    print(f"Actual Catheter Angle (with noise): {catheter_angle:.2f} degrees")
    print(f"Updated Catheter Length: {length_c:.3f} meters")
    print(f"Magnet Rotation Increment: {rotation_adjustment:.2f} radians")


    catheter_angles.append(catheter_angle)
    catheter_lengths.append(length_c)
    magnet_rotations.append(np.degrees(rotation_adjustment))

    time.sleep(0.1)  


plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(time_points, target_angles, 'b--', label="Target Angle (degrees)")
plt.plot(time_points, catheter_angles, 'r-', label="Actual Catheter Angle (degrees)")
plt.xlabel("Time (seconds)")
plt.ylabel("Angle (degrees)")
plt.title("Catheter Angle Tracking")
plt.legend()
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(time_points, catheter_lengths, 'g-', label="Catheter Length (meters)")
plt.xlabel("Time (seconds)")
plt.ylabel("Length (m)")
plt.title("Catheter Length Over Time")
plt.legend()
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(time_points, magnet_rotations, 'm-', label="Magnet Rotation (degrees)")
plt.xlabel("Time (seconds)")
plt.ylabel("Rotation (degrees)")
plt.title("Incremental Magnet Rotation Adjustments")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
