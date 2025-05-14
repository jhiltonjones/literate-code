import numpy as np

spring_mass_params = {
    'm': 1.0,        # kg
    'k': 10.0,       # N/m
    'c': 1.0,        # Ns/m
    'Ts': 0.1        # timestep
}

def spring_mass_update(t, x, u, params):
    # x = [position, velocity]
    m, k, c, Ts = params['m'], params['k'], params['c'], params['Ts']
    pos, vel = x

    # dynamics
    acc = (-k * pos - c * vel + u[0]) / m
    vel_next = vel + acc * Ts
    pos_next = pos + vel_next * Ts

    return np.array([pos_next, vel_next])
