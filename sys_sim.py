import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import control as ct
import numpy as np
from catheter_system import catheter_sys, catheter_update, catheter_params
# Time vector
timepts = np.arange(0, 11, 1.0)  # Steps: 0, 1, 2, ..., 10

# Input U: shape must be (3, len(timepts)) for 3 inputs [magnet_x, magnet_y, magnet_angle]
# Let's apply a rotating magnetic field that moves forward slowly
magnet_x = 0.05 * np.ones_like(timepts)
magnet_y = 0.01 * np.sin(0.5 * timepts)
magnet_angle = 0.1 * np.sin(5 * timepts)

U = np.vstack([magnet_x, magnet_y, magnet_angle])
# Initial state: [tip_x, tip_y, base_x, base_y]
x0 = np.array([0.0, 0.0, -0.03, 0.0])  # tip is ahead of base
resp = ct.input_output_response(catheter_sys, timepts, U, x0)
# Using method call
resp.plot()
plt.show()
t, y, x = resp.time, resp.outputs, resp.states
