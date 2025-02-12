import numpy as np
from constants import *
from magnetic import *
from magnetic import volume_calculator_cyclinder, magnetic_moment, magnetic_field_external_magnet
def moment_cath1(x_basis, y_basis, alpha_o, theta_l):
    moment_cath_unit = np.cos(theta_l)*x_basis + np.sin(theta_l) * y_basis
    moment_mag_unit = np.cos(alpha_o)*x_basis - np.sin(alpha_o) * y_basis
    return moment_cath_unit, moment_mag_unit

def compute_center_of_catheter(l_c, kappa, theta_l):
    x_l = (1 / kappa) * np.sin(theta_l)
    y_l = (1 / kappa) * (1 - np.cos(theta_l))

    x_c = x_l + ((l_c / 2) * np.cos(theta_l))
    y_c = y_l + ((l_c / 2) * np.sin(theta_l))

    return np.array([x_c, y_c])

def compute_unit_position_vector(x_c, y_c, d, h):


    p_norm1 = np.sqrt((x_c - d)**2 + (y_c - h)**2)
    x_p = (x_c - d) / p_norm1
    y_p = (y_c - h) / p_norm1
    return np.array([x_p, y_p, 0]), p_norm1, y_p, x_p

# def compute_magnetic_torque(mu_0, m_a, m_c, p_hat):

#     p_norm = np.linalg.norm(p_hat)
#     n_0 = (mu_0 * np.linalg.norm(m_a) * np.linalg.norm(m_c)) / (4 * np.pi * p_norm**3)

#     identity_matrix = np.eye(3)
#     interaction_matrix = 3 * np.outer(p_hat, p_hat) - identity_matrix

#     T_m = n_0 * np.cross(m_c, interaction_matrix @ m_a)
#     return T_m

def components(x_p, y_p):

    n1 = (3 * x_p**2) - 1
    n2 = -3 * x_p * y_p
    n3 = -n2  
    n4 = 1 - (3 * y_p**2)
    
    return n1, n2, n3, n4


def compute_T_m(n0, theta_l, alpha_o, n1, n2, n3, n4):

    T_z = n0 * ((np.cos(theta_l) * n3 - np.sin(theta_l) * n1) * np.cos(alpha_o) + 
                (np.cos(theta_l) * n4 - np.sin(theta_l) * n2) * np.sin(alpha_o))
    # print(f"n0: {n0}, n1: {n1}, n2: {n2}, n3: {n3}, n4: {n4}")
    # print(f"Torque Calculation: T_z = {T_z}")

    return T_z 

def compute_f1_f2(n0, theta_l, n1, n2, n3, n4):

    f1 = n0 * (np.cos(theta_l) * n3 - np.sin(theta_l) * n1)
    f2 = n0 * (np.cos(theta_l) * n4 - np.sin(theta_l) * n2)

    return f1, f2

def compute_alpha_o_star(EI, theta_l, l, f1, f2):

    term1 = np.arcsin((EI * theta_l) / (l * np.sqrt(f1**2 + f2**2)))
    term2 = np.arctan2(f1, f2) 

    return term1 - term2

if __name__ == "__main__":
    x_c, y_c = compute_center_of_catheter(length_c, 0.84, theta_l)
    print(f"Center of Catheter: x_c = {x_c}, y_c = {y_c}")

    Volume_catheter = volume_calculator_cyclinder(s_c, length_c)
    Volume_magnet = volume_calculator_cyclinder(s_a, h_a)

    # p_hat = compute_unit_position_vector(x_c, y_c, d, h)
    print("Center of catheter is: ", x_c, y_c)
    # print(f"Unit Position Vector: {p_hat}")
    # moment_cath_unit, moment_mag_unit = moment_cath1(x_distance, y_distance)
    # B = magnetic_field_external_magnet(mu_0, p_hat, moment_mag_unit)
    # x_p, y_p, _ = p_hat
    # p_norm = np.linalg.norm(p_hat)
    # n0 = (mu_0 * moment_mag_unit * moment_cath_unit) / (4 * np.pi * p_norm**3)
    p_hat, p_norm1, y_p, x_p = compute_unit_position_vector(x_c, y_c, d, h)

    n1, n2, n3, n4 = components(x_p, y_p)
    # T_m = compute_T_m(n0, theta_l, alpha_o, n1, n2, n3, n4)
    # print("Magnetic Torque:", T_m)

    # # Compute f1, f2
    # f1, f2 = compute_f1_f2(n0, theta_l, n1, n2, n3, n4)
    # print("f1:", f1, "f2:", f2)

    # # Compute optimal α_o*
    # alpha_o_star = compute_alpha_o_star(EI, theta_l, length, f1, f2)
    # print("Optimal α_o* (degrees):", np.degrees(alpha_o_star))