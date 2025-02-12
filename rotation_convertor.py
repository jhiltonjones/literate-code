import numpy as np

# Given rotation differences in radians
delta_rx = -0.2
delta_ry = 0.6
delta_rz = 0.4

# Convert to degrees
delta_rx_deg = np.degrees(delta_rx)
delta_ry_deg = np.degrees(delta_ry)
delta_rz_deg = np.degrees(delta_rz)
rad1 = np.deg2rad(-20)
rad2 = np.deg2rad(-120)
rad3 = np.deg2rad(0)
# Output results
print(delta_rx_deg, delta_ry_deg, delta_rz_deg)
# print(rad1, rad2, rad3)
