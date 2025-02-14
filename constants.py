import math
import numpy as np
mu_0 = 4* math.pi * 1e-7
# length = 0.0192 #m
# length_m = 0.007 #m length of marker 2
length_c_m = 0.006
length_c = 0.0192 #m legnth of magnetic catheter
s_d = 0.0025 #m diamter of marker 1
s_c = 0.001 #m diameter of catheter
EI  = 2.14e-6 #N*m**2 youngs modulus
B_ra = 1.35 #T Residual flux density of external magnet
B_rc = 1.44 #T Residual flux density fo magnetic catheter
s_a = 0.025
h_a = 0.05 #m diameter and height of of the driving magent
d_off = 0.099 #m offset distance between {E} and the flange
v_r = 0.006 #m/s insertion speed
epsilon = 0.0012 #m detection threshold
I = np.eye(3)
# theta_l =np.radians(75.76)
theta_l =np.radians(-25.88)
# theta_l =np.radians(-53.91)
theta_f = np.pi / 6
# p = [0,0.05,0.05]
# p_norm = np.linalg.norm(p)
# m_c_hat = np.array([1,0,0])
# m_a_hat = np.array([0,0,1])
# d, h = 0.086, 0.0
d, h = 0.086, -0.025
# d, h = 0.086, 0.025
y_basis = np.array([0,1,0])
x_basis = np.array([1,0,0])
t_basis = np.array([1,1,0])
def degrees_to_radians(degrees):
    result = degrees * (np.pi/180)
    return result
def radians_to_degrees(radians):
    degrees = radians/(np.pi/180)
    return degrees
def theta_plan(x_basis, t_basis):
    term1 = x_basis @ t_basis
    term2 = np.linalg.norm(x_basis)*np.linalg.norm(t_basis)
    result = np.arccos(term1 / term2)
    return result
if __name__ == "__main__":
    theta = theta_plan(x_basis, t_basis)
    print("Theta is: ", theta)
    answer_deg = radians_to_degrees(theta)
    print("Answer in degrees is: ", answer_deg)
    # answer_rad = degrees_to_radians(-53.91)
    # print("In radians this is: ", answer_rad)


