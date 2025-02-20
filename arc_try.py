import sys
import logging
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config
import time
import numpy as np

def setp_to_list(setp, offset=0):
    return [setp.__dict__[f"input_double_register_{i + offset}"] for i in range(6)]


def list_to_setp(setp, lst, offset=0):
    """ Converts a list into RTDE input registers, allowing different sets with offsets. """
    for i in range(6):
        setp.__dict__[f"input_double_register_{i + offset}"] = lst[i]
    return setp

# ------------- Robot Communication Setup -----------------
ROBOT_HOST = '192.168.56.101'
ROBOT_PORT = 30004
config_filename = 'control_loop_configuration.xml'  

logging.getLogger().setLevel(logging.INFO)

conf = rtde_config.ConfigFile(config_filename)
state_names, state_types = conf.get_recipe('state')  
setp_names, setp_types = conf.get_recipe('setp')  
watchdog_names, watchdog_types = conf.get_recipe('watchdog')


con = rtde.RTDE(ROBOT_HOST, ROBOT_PORT)
connection_state = con.connect()

while connection_state != 0:
    time.sleep(0.5)
    connection_state = con.connect()
print("---------------Successfully connected to the robot-------------\n")

con.get_controller_version()


FREQUENCY = 500  
con.send_output_setup(state_names, state_types, FREQUENCY)
setp = con.send_input_setup(setp_names, setp_types)
watchdog = con.send_input_setup(watchdog_names, watchdog_types)

for i in range(12): 
    setattr(setp, f"input_double_register_{i}", 0)

# setp.input_double_register_0 = 0
# setp.input_double_register_1 = 0
# setp.input_double_register_2 = 0
# setp.input_double_register_3 = 0
# setp.input_double_register_4 = 0
# setp.input_double_register_5 = 0
setp.input_bit_registers0_to_31 = 0
watchdog.input_int_register_0 = 0


if not con.send_start():
    sys.exit()


start_pose = [-0.1636516157357605, 0.689378623744684, 0.6082692827751903, 3.0881598934120515, -0.5502563299347971, 0.006655673208488258]
while True:
    # print('Boolean 1 is False, please click CONTINUE on the Polyscope')
    state = con.receive()
    con.send(watchdog)
    if state.output_bit_registers0_to_31:  
        print('Boolean 1 is True, proceeding to mode 1\n')
        break

watchdog.input_int_register_0 = 1
con.send(watchdog)  # sending mode == 1
list_to_setp(setp, start_pose)  # changing initial pose to setp
con.send(setp) # sending initial pose

while True:
    print('Waiting for movej() to finish')
    state = con.receive()
    con.send(watchdog)
    if state.output_bit_registers0_to_31 == False:
        print('Proceeding to mode 2\n')
        break

# Move Up and Adjust Orientation to Ensure End-Effector Z-Axis Aligns with Y-Axis
height_increase = 0.2  # Increase the height by 0.1m
new_z = start_pose[2] + height_increase

# Define the new center position for the arc
center = [-0.09754984739093804, 0.6522905275701674, 0.35294565469188244]

# Compute Vector to Center
dx = center[0] - start_pose[0]
dy = center[1] - start_pose[1]
dz = center[2] - new_z

distance_to_center = np.sqrt(dx**2 + dy**2 + dz**2)

# Ensure the Z-axis of the end-effector is aligned with the Y-axis
rx = np.pi/2   # Rotate around X-axis by 90 degrees to align Z-axis with Y-axis
ry = 0           # No rotation around Y-axis
rz = 0           # No rotation around Z-axis

# Define New Target Pose
target_pose = [start_pose[0], start_pose[1], new_z, rx, ry, rz]

# Send Move Command
print("Executing move up and reorientation...")
watchdog.input_int_register_0 = 2
con.send(watchdog)
list_to_setp(setp, target_pose)
con.send(setp)
time.sleep(2)

while True:
    state = con.receive()
    con.send(watchdog)
    if not state.output_bit_registers0_to_31:
        break

print("✅ Move up and reorientation completed!")

watchdog.input_int_register_0 = 3
con.send(watchdog)

con.send_pause()
con.disconnect()
print("Disconnected")

