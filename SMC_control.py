import numpy as np
import matplotlib.pyplot as plt


dt = 0.01  
T = 5  
num_steps = int(T / dt)


m = 0.1  
c1, c2 = 5, 1  
alpha, beta = 10, 5  
b = 0.1  


x_ref = lambda t: np.sin(0.5 * t)  


x = 0.0  
x_dot = 0.0
s_int = 0.0  


time = np.arange(0, T, dt)
x_traj = np.zeros(num_steps)
x_ref_traj = np.zeros(num_steps)
control_input = np.zeros(num_steps)

# Saturation function
def sat(s, b):
    return np.clip(s / b, -1, 1)


for i in range(num_steps):
    t = i * dt  
    
    e = x_ref(t) - x 
    e_dot = -x_dot  
    
    s_int += e * dt

    s = c1 * e + c2 * s_int

    u = -alpha * s - beta * sat(s, b)
    
    x_ddot = u / m 
    x_dot += x_ddot * dt  
    x += x_dot * dt  

    # Store data
    x_traj[i] = x
    x_ref_traj[i] = x_ref(t)
    control_input[i] = u

