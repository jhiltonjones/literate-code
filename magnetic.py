import pandas as pd
import math
import constants
import numpy as np



def magnetic_field_external_magnet(mu_0, p, m_a):

    p_norm = np.linalg.norm(p)
    if p_norm == 0:
        raise ValueError("Position vector p cannot be zero.")
    p_hat = p / p_norm  

    mu_0_factor = 1e-7  
    term1 = (3 * np.dot(m_a, p_hat) * p_hat - m_a) / (p_norm**3)
    B = mu_0_factor * term1

    return np.array(B) 
def compute_torque(c_vector, m_vector, p, n_0):
    term1 = n_0 * c_vector
    term2 = ((3 * p * p.T) - np.eye(3)) * m_vector
    output = np.cross(term1, term2)
    return output

def volume_calculator_cyclinder(radius, height):
    volume = np.pi * radius**2 * height
    return volume

def magnetic_moment(mu_0, B, Volume, m_hat):
    magnetic_moment_v = (B * Volume * m_hat) / mu_0  # Correct scaling
    moment_vector = np.linalg.norm(magnetic_moment_v)
    return magnetic_moment_v, moment_vector


def unit_vector(m):
    norm_m = np.linalg.norm(m)
    if norm_m ==0:
        raise ValueError("Zero vector has no unit vector.")  
    return m / norm_m  

def magnetic_field_gradient(mu_0, p, M, delta=1e-5):
    grad_B = np.zeros((3, 3)) 

    for i in range(3):  
        p_shifted = p.copy() 

        p_shifted[i] += delta  
        B_plus = magnetic_field_external_magnet(mu_0, p_shifted, M)  

        p_shifted[i] -= 2 * delta  
        B_minus = magnetic_field_external_magnet(mu_0, p_shifted, M)  
        grad_B[:, i] = (B_plus - B_minus) / (2 * delta)  
    return grad_B  

def magnetic_force(mu_0, p, M_ext, M_catheter):
    grad_B = magnetic_field_gradient(mu_0, p, M_ext)
    F_m = grad_B.dot(M_catheter)
    return F_m
def magnetic_torque(M_catheter, B):
    return np.cross(M_catheter, B)
