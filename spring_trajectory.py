import numpy as np

def compute_desired_trajectory_spring(k, N, step, Ts=1.0):
    """
    Generates a smooth sinusoidal desired trajectory for a spring-mass system.
    For 1D mass: only x-axis is used. For 2D mass, x is sinusoidal, y is linear.
    """
    t = np.arange(k, k + N) * Ts
    amplitude = 1.0        # meters
    frequency = 0.1        # Hz
    velocity_y = 0.1       # m/s

    x = amplitude * np.sin(2 * np.pi * frequency * t)
    y = velocity_y * t

    # for 1D spring-mass, you can just return x as position
    # for 2D or if you want consistency with catheter shape (2D array):
    path = np.vstack((x, y)).T   # shape (N, 2)

    # suggested velocity is only for reference, not used in spring-mass
    v_suggested = np.mean(np.sqrt((2 * np.pi * frequency * amplitude * np.cos(2 * np.pi * frequency * t))**2 + velocity_y**2))

    return path, v_suggested
