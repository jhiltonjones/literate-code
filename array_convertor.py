# Re-import necessary libraries since execution state was reset
import numpy as np

# Given angles in radians
angles_radians = np.array([-0.31561346376380156, 0.4392848297605487, 0.3870643751382633, -3.1129835149573597, -0.39044182826995194, -0.02877771799883438])
# Convert to degrees
angles_degrees = np.degrees(angles_radians)

# Output results
print(angles_degrees)
