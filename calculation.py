import numpy as np

catheter_tip = np.array([-0.08307263730334469, 0.4518397120336689, 0.19327695477195206])
magnet = np.array([-0.2123690377862869, 0.44448146207089445, 0.20914216190632756])
start =  [-1.482638184224264, -1.535812971448042, 2.513336483632223, 3.7579075533100585, -1.5675271193133753, -0.4420364538775843]
end =[-1.4826105276690882, -1.5357781325331707, 2.5133329073535364, 3.757891817683838, -1.5675237814532679, 0.5487656593322754]
# Compute the direction vector
direction_vector = catheter_tip - magnet
joint_correct = start[5] - end[5]
# Normalize the vectors
catheter_tip_norm = np.linalg.norm(catheter_tip)
magnet_norm = np.linalg.norm(magnet)
direction_norm = np.linalg.norm(direction_vector)

# Compute the angle
angle = np.arccos(np.dot(catheter_tip, magnet) / (catheter_tip_norm * magnet_norm))
angle_direction = np.arccos(np.dot(direction_vector, magnet) / (direction_norm * magnet_norm))
print(f'Joint correct = {joint_correct}')
print(f"x distance between vecotrs: {abs(magnet[0] - catheter_tip[0])*100} mm")
print("Angle between position vectors:", np.rad2deg(angle))
print("Angle using direction vector:", np.rad2deg(angle_direction))
