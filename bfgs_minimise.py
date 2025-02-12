import numpy as np
from scipy.optimize import minimize
from ik_modelling import compute_unit_position_vector, components, moment_cath1, compute_center_of_catheter, compute_T_m
from constants import *
from kinematic import bending_moment_equation, compute_curvature, function_alpha_o, force_12
from magnetic import compute_torque, magnetic_moment, volume_calculator_cyclinder, magnetic_field_external_magnet

import numpy as np

def compute_angle(EI, theta_l, length_c, x_p, y_p, x_basis, y_basis, p_norm1, 
                  max_iterations=100, tolerance=1e-10):
    alpha_star_alternate = np.radians(5)  # Initial guess

    for i in range(max_iterations):
        print(f"\n🔵 **Iteration {i+1}** 🔵")
        
        # Compute catheter and magnet unit vectors
        vector_cath, vector_mag = moment_cath1(x_basis, y_basis, alpha_star_alternate, theta_l)
        print(f"Vector Catheter: {vector_cath}")
        print(f"Vector Magnet: {vector_mag}")

        # Compute magnetic moments
        moment_cath, _ = magnetic_moment(mu_0, B_rc, volume_cath, vector_cath)
        moment_mag, _ = magnetic_moment(mu_0, B_ra, volume_mag, vector_mag)
        print(f"Magnetic Moment Catheter: {moment_cath}")
        print(f"Magnetic Moment Magnet: {moment_mag}")

        # Compute norm of magnetic moments
        moment_mag_unit_norm = np.linalg.norm(moment_mag)
        moment_cath_unit_norm = np.linalg.norm(moment_cath)
        print(f"Norm Magnetic Moment Magnet: {moment_mag_unit_norm}")
        print(f"Norm Magnetic Moment Catheter: {moment_cath_unit_norm}")

        # Compute n0 (magnetic interaction scaling factor)
        n0 = (mu_0 * moment_mag_unit_norm * moment_cath_unit_norm) / (4 * np.pi * p_norm1**3)
        print(f"n0: {n0}")

        # Compute force components
        n1, n2, n3, n4 = components(x_p, y_p)
        print(f"n1: {n1}, n2: {n2}, n3: {n3}, n4: {n4}")

        f_1, f_2 = force_12(n0, n1, n2, n3, n4, theta_l)
        print(f"Computed Forces → f1: {f_1}, f2: {f_2}")

        # Compute bending moments
        T_e, T_e2 = bending_moment_equation(EI, theta_l, length_c, f_1, f_2, alpha_star_alternate)
        print(f"EI * theta_l / l: {T_e}")
        print(f"Bending from forces: {T_e2}")
        print(f"Difference: {T_e - T_e2}")

        # Compute new alpha_star_alternate
        new_alpha_star_alternate = function_alpha_o(EI, theta_l, f_1, f_2, length_c)
        print(f"Function Alpha_o* Term1: {new_alpha_star_alternate}")

        # Print debug information for angle update
        print(f"Iteration {i+1}: α_o = {np.degrees(alpha_star_alternate):.6f}° → New α_o = {np.degrees(new_alpha_star_alternate):.6f}°")

        # Check for convergence
        if np.abs(new_alpha_star_alternate - alpha_star_alternate) < tolerance:
            print(f"✅ Converged in {i+1} iterations.")
            break

        # Update α_o for next iteration
        alpha_star_alternate = new_alpha_star_alternate

    return alpha_star_alternate




def objective_function(alpha_o, EI, theta_l, length_c, x_p, y_p, x_basis, y_basis, p_norm1):
    # B = magnetic_field_external_magnet(mu_0, p_norm1, B_ra)
    # print(B)
    vector_cath, vector_mag = moment_cath1(x_basis, y_basis, alpha_o, theta_l)
    print("Vector catheter: ", vector_cath)
    print("Vector magnet: ", vector_mag)

    # Compute magnetic moments
    moment_cath, _ = magnetic_moment(mu_0, B_rc, volume_cath, vector_cath)
    moment_mag, _ = magnetic_moment(mu_0, B_ra, volume_mag, vector_mag)
    print(f"Magnetic Moment Catheter: {moment_cath}")
    print(f"Magnetic Moment Magnet: {moment_mag}")

    # Compute norm of magnetic moments
    moment_mag_unit_norm = np.linalg.norm(moment_mag)
    moment_cath_unit_norm = np.linalg.norm(moment_cath)
    print(f"Norm Magnetic Moment Magnet: {moment_mag_unit_norm}")
    print(f"Norm Magnetic Moment Catheter: {moment_cath_unit_norm}")

    # Compute n0 (magnetic interaction scaling factor)
    n0 = (mu_0 * moment_mag_unit_norm * moment_cath_unit_norm) / (4 * np.pi * p_norm1**3)
    print(f"n0: {n0}")
    n1, n2, n3, n4 = components(x_p, y_p)
    print(f"n1: {n1}, n2: {n2}, n3: {n3}, n4: {n4}")
    T_m= compute_T_m(n0, theta_l, alpha_o, n1, n2, n3, n4)  
    
    # torque = compute_torque(vector_cath, vector_mag, p_unit, n0)
    print("Torque: ", T_m)
    # T_m = T_m_vector[2]
    # T_e = (EI * theta_l) / length_c  
    f_1, f_2 = force_12(n0, n1, n2, n3, n4, theta_l)
    print(f"Computed Forces → f1: {f_1}, f2: {f_2}")
    # print(f"f1: {f_1}, f2: {f_2}")
    # alpha_star_alternate = function_alpha_o(EI, theta_l, f_1, f_2, length_c)
    # print("Alpha star alternate is: ", alpha_star_alternate)
    T_e, T_e2 = bending_moment_equation(EI, theta_l, length_c, f_1, f_2, alpha_o)
    # print("Bending Moment2: ", T_e2)
    # print("Bending Moment: ", T_e)
    print(f"EI * theta_l / l: {T_e}")
    print(f"Bending from forces: {T_e2}")
    print(f"Difference: {T_e - T_e2}")
    residual = (T_e - T_m)
    alpha_o_scalar = alpha_o.item()
    # print(f"Evaluating α_o = {np.degrees(alpha_o_scalar):.3f}° → Residual Norm = {np.linalg.norm(residual):.6f}")
    # print("The torque is: ", T_m)


    return np.linalg.norm(residual)

def find_optimal_alpha_o(EI, theta_l, length_c, x_p, y_p, x_distance, y_distance, p_norm1):
    initial_guesses = [np.radians(5), np.radians(90), np.radians(150)]  # Try different angles
    best_result = None
    best_residual = np.inf

    for guess in initial_guesses:
        result = minimize(objective_function, guess, args=(EI, theta_l, length_c, x_p, y_p, x_distance, y_distance, p_norm1), method='BFGS')
        
        if result.fun < best_residual:  # Keep the best solution
            best_residual = result.fun
            best_result = result

    return best_result.x[0]

  
# if __name__ == "__main__":
kappa = compute_curvature(theta_l, length_c)
print(f"Curvature kappa: {kappa}")
volume_cath = volume_calculator_cyclinder((s_c/2), length_c_m)
volume_mag = volume_calculator_cyclinder((h_a/2), h_a)
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
print("P_unit vector: ", p_unit)
print("Actual Distance p_norm:", p_norm1)  
print("Unit Vector p_hat:", p_hat)
# p_norm = np.linalg.norm(p_hat)
print("Distance: ", p_norm1)
# alpha_o_star = find_optimal_alpha_o(EI, theta_l, length_c, x_p, y_p, x_basis, y_basis, p_norm1)
# print(f"Optimal α_o* (degrees): {np.degrees(alpha_o_star):.3f}°")
alpha_star = compute_angle(EI, theta_l, length_c, x_p, y_p, x_basis, y_basis, p_norm1)
alpha_star_deg = np.rad2deg(alpha_star)
print("Final Alpha is: ", alpha_star_deg)