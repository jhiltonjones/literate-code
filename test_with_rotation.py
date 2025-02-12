import sys
import logging
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config
import time
import numpy as np
import matplotlib.pyplot as plt
from min_jerk_test import PathPlanTranslation  # Ensure this module is available

# -------- functions -------------
def setp_to_list(setp):
    return [setp.__dict__["input_double_register_%i" % i] for i in range(6)]

def list_to_setp(setp, lst):
    for i in range(6):
        setp.__dict__["input_double_register_%i" % i] = lst[i]
    return setp

# ------------- Robot Communication Setup -----------------
ROBOT_HOST = '192.168.56.101'
ROBOT_PORT = 30004
config_filename = 'control_loop_configuration.xml'  # specify XML file for data synchronization

logging.getLogger().setLevel(logging.INFO)

conf = rtde_config.ConfigFile(config_filename)
state_names, state_types = conf.get_recipe('state')  
setp_names, setp_types = conf.get_recipe('setp')  
watchdog_names, watchdog_types = conf.get_recipe('watchdog')

# Establish Connection
con = rtde.RTDE(ROBOT_HOST, ROBOT_PORT)
connection_state = con.connect()

while connection_state != 0:
    time.sleep(0.5)
    connection_state = con.connect()
print("---------------Successfully connected to the robot-------------\n")

con.get_controller_version()

# Setup Data Synchronization
FREQUENCY = 500  
con.send_output_setup(state_names, state_types, FREQUENCY)
setp = con.send_input_setup(setp_names, setp_types)
watchdog = con.send_input_setup(watchdog_names, watchdog_types)

# Initialize Registers
for i in range(6):
    setattr(setp, f"input_double_register_{i}", 0)

setp.input_bit_registers0_to_31 = 0
watchdog.input_int_register_0 = 0

# Start Data Synchronization
if not con.send_start():
    sys.exit()

# Define Waypoints for Multi-Point Trajectory
waypoints = [
    [-0.32, 0.38, 0.30, 3.1, -0.5, 0.0],
    [-0.32, 0.38, 0.30, 3.1, -0.5, 0.0]
    # [-0.404, 0.516, 0.4, 3.1, -0.5, 0.0]  # Start
    # [-0.119, 0.516, 0.2, 3.1, -0.7, 0.0]    # Intermediate 1
    # [-0.119, 0.616, 0.2, 3.1, -0.5, 0.0],    # Intermediate 2
    # [-0.404, 0.616, 0.2, 3.1, -0.5, 0.0],      # Final
    # [-0.404, 0.716, 0.2, 3.1, -0.5, 0.0],
    # [-0.119, 0.716, 0.2, 3.1, -0.5, 0.0]
]

trajectory_time = 5
trajectory_time_per_segment = trajectory_time / (len(waypoints) - 1)  
# orientation_const = waypoints[0][3:]

# Wait for User to Continue on Polyscope
while True:
    print('Boolean 1 is False, please click CONTINUE on the Polyscope')
    state = con.receive()
    con.send(watchdog)
    if state.output_bit_registers0_to_31:  
        print('Boolean 1 is True, proceeding to mode 1\n')
        break

print("-------Executing moveJ -----------\n")

# Send Initial Pose
watchdog.input_int_register_0 = 1
con.send(watchdog)
list_to_setp(setp, waypoints[0])  
con.send(setp)

while True:
    print('Waiting for moveJ() to finish')
    state = con.receive()
    con.send(watchdog)
    if not state.output_bit_registers0_to_31:
        print('Proceeding to mode 2\n')
        break

# Execute ServoJ with Multi-Waypoint Trajectory
print("-------Executing servoJ  -----------\n")
watchdog.input_int_register_0 = 2
con.send(watchdog)

plotter = True
time_plot, min_jerk_x, min_jerk_y, min_jerk_z = [], [], [], []
min_jerk_rx, min_jerk_ry, min_jerk_rz = [], [], []
px, py, pz, rx, ry, rz = [], [], [], [], [], []

# Multi-Waypoint Execution
current_waypoint_index = 0
t_start = time.time()

while current_waypoint_index < len(waypoints) - 1:
    t_segment_start = time.time()
    planner = PathPlanTranslation(waypoints[current_waypoint_index], waypoints[current_waypoint_index + 1], trajectory_time_per_segment)

    while time.time() - t_segment_start < trajectory_time_per_segment:
        t_current = time.time() - t_segment_start
        # print(f"t_current: {t_current}, trajectory_time_per_segment: {trajectory_time_per_segment}")

        state = con.receive()

        if state.runtime_state > 1:
            position_ref, rotation_ref, _, _, _, _ = planner.trajectory_planning(t_current)
            # print(f"Velocity reference: {lin_vel_ref}")


            pose = np.concatenate((position_ref, rotation_ref))  # Combine position & rotation
            list_to_setp(setp, pose)
            con.send(setp)

            if plotter:
                time_plot.append(time.time() - t_start)
                min_jerk_x.append(position_ref[0])
                min_jerk_y.append(position_ref[1])
                min_jerk_z.append(position_ref[2])
                min_jerk_rx.append(rotation_ref[0])
                min_jerk_ry.append(rotation_ref[1])
                min_jerk_rz.append(rotation_ref[2])

                px.append(state.actual_TCP_pose[0])
                py.append(state.actual_TCP_pose[1])
                pz.append(state.actual_TCP_pose[2])
                rx.append(state.actual_TCP_pose[3])
                ry.append(state.actual_TCP_pose[4])
                rz.append(state.actual_TCP_pose[5])

    current_waypoint_index += 1  

print(f"It took {time.time()-t_start}s to execute the servoJ")

state = con.receive()


# Mode 3
watchdog.input_int_register_0 = 3
con.send(watchdog)

con.send_pause()
con.disconnect()

# Plot Results
if plotter:
    plt.figure()
    plt.plot(time_plot, min_jerk_x, label="x_min_jerk")
    plt
