import numpy as np

robot_arm_start = np.array([0.715779247419866, 0.263790306738413, 0.6893919061406618, 2.9815894477810647, 0.9528468882010935, 0.02809155233607956])
catheter_start = np.array([(0.7054119710072467- 0.05), 0.3477044989491751, 0.600535658177751, 2.884200185793592, 1.1180416619550402, -0.058065734325369])

angle_norm = np.linalg.norm(robot_arm_start - catheter_start)
z_height = robot_arm_start[2] - catheter_start[2]
x_axis = robot_arm_start[0] - catheter_start[0]
y_axis = robot_arm_start[1] - catheter_start[1]
print(f"The euclidean distance is {angle_norm}")
print(f"The height in the z direction is {z_height}")
print(f"The difference in x is {x_axis}")
print(f"The difference in the y axis is {y_axis}")
output = 20.1 / np.pi
print(output)
