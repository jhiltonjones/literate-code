import math 
import numpy as np
from constants import *
# from magnetic import *
import scipy.integrate as spi

def force_12(n0, n1, n2, n3, n4, theta_l):
    f_1 = n0*((np.cos(theta_l)*n3) - (np.sin(theta_l)*n1))
    f_2 = n0*((np.cos(theta_l)*n4) - (np.sin(theta_l)*n2))
    return f_1, f_2
def compute_bending(f_1, f_2, alpha_o):
    output = (f_1 * np.cos(alpha_o)) + (f_2*np.sin(alpha_o))
    return output
def bending_moment_equation(EI, theta_l, length_c, f_1, f_2, alpha_o):
    """ Computes EI * theta_l / l and compares it with the computed bending force. """
    
    # Left-hand side of equation
    T_e = (EI * theta_l) / length_c  
    
    # Right-hand side (force-based bending)
    T_e2 = compute_bending(f_1, f_2, alpha_o)
    
    return T_e, T_e2
def function_alpha_o(EI, theta_l, f_1, f_2, l):
    term1 = (EI * theta_l) / (np.sqrt(f_1**2 + (f_2**2)) * l)
    print("Term1: ", term1)
    term1 = np.clip(term1, -1, 1)  # Ensure arcsin input is valid
    term2 = np.arcsin(term1)

    # Use arctan2 to ensure correct quadrant
    term3 = np.arctan2(f_1, f_2)

    output = term2 - term3
    return output

def compute_bending_moment_l(l_c, theta_F, theta_l, F_m, T_m):
    return (l_c * np.linalg.norm(F_m) * np.sin(theta_F - theta_l) + 2 * np.linalg.norm(T_m)) / 2

# def integrand(theta):
#     term = (M_l**2 / (2 * EI)) + Fy * np.sin(theta) + Fx * np.cos(theta)
#     return term**(-0.5)

# def compute_deflection(theta_max):
#     integral_result, _ = spi.quad(integrand, 0, theta_max)
#     x = np.sqrt(EI / 2) * integral_result * np.cos(theta_max)
#     y = np.sqrt(EI / 2) * integral_result * np.sin(theta_max)
#     return x, y
def compute_curvature(theta_l, length_c):
    return np.abs(theta_l) / length_c  


# Volume_catheter = volume_calculator_cyclinder(s_d, length_c)
# M_catheter, M_catheter_scaler = magneitc_moment(mu_0, B_ra, Volume_catheter, m_c_hat)
# Volume_magnet = volume_calculator_cyclinder(s_a, h_a)
# M_mag, M_mag_scaler = magneitc_moment(mu_0, B_ra, Volume_magnet, m_a_hat)
# B = magnetic_field_external_magnet(mu_0, p_hat, M_mag_scaler)
# B_norm = np.linalg.norm(B)*1e3
# T_m = magnetic_torque(M_catheter, B)
# F_m = magnetic_force(mu_0, p, M_mag, M_catheter)
# M_l = compute_bending_moment_l(length_c, theta_f, theta_l, F_m, T_m)
# Fx = np.linalg.norm(F_m) * np.sin(theta_l)
# Fy = np.linalg.norm(F_m) * np.cos(theta_l)
# C = C = (M_l**2) / (2 * EI) + Fy * np.sin(theta_l) + Fx * np.cos(theta_l)
# print("Volume of Catheter: ", Volume_catheter)
# print("Magnetic moment of Catheter:", M_catheter_scaler)
# print("Magnetic moment of Magnet:", M_mag_scaler)
# print("Magnetic field: ", B)
# print("Magnetic field norm (mT): ", B_norm)
# print("Torque: ", T_m)
# print("Force: ", F_m)


