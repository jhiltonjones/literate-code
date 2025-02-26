import cvxpy as cp
import numpy as np
import time

def mpc(target, actual):
    
    N = 10  # Prediction horizon
    A = 0.9  # Catheter angle retention factor 
    B = 0.05  # Magnet rotation effect on catheter bending
    lambda_reg = 0.1  # Regularization for smooth control
    u_max = 10  

    alpha_star_opt = cp.Variable(N)  
    theta_catheter = cp.Variable(N+1)  

    theta_init = actual  

    cost = 0
    constraints = [theta_catheter[0] == theta_init]  

    for i in range(N):
        constraints += [theta_catheter[i+1] == A * theta_catheter[i] + B * alpha_star_opt[i]]
        
        cost += cp.square(theta_catheter[i] - target) + lambda_reg * cp.square(alpha_star_opt[i])

        constraints += [cp.abs(alpha_star_opt[i]) <= u_max]

    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve()

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        print("Warning: MPC solver failed, returning 0 rotation.")
        return 0  

    optimal_alpha_star = alpha_star_opt.value[0] 

    optimal_alpha_star = np.clip(optimal_alpha_star, -u_max, u_max)

    return optimal_alpha_star


if __name__ == "__main__":
    vessel_branch_target_angle = 45
    catheter_tip_position = 40
    optimal_magnet_rotation =   mpc(vessel_branch_target_angle, catheter_tip_position)
    print(optimal_magnet_rotation)