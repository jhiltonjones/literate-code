import numpy as np

catheter_tip = np.array([-0.08307263730334469, 0.4518397120336689, 0.19327695477195206])
magnet = np.array([-0.2123690377862869, 0.44448146207089445, 0.20914216190632756])
start =  [-1.4825952688800257, -1.55037002012644, 2.5089641253100794, 3.776845617885254, -1.5674737135516565, -0.6004284063922327]
end =[-1.4825952688800257, -1.5503355108997603, 2.5089243094073694, 3.7768851953693847, -1.567427937184469, 0.15478314459323883]
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
