import sys
import logging
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config
import time
import numpy as np
import matplotlib.pyplot as plt
from trajectory_planner_line import PathPlanTranslation  # Ensure this module is available
from bfgs_minimise import alpha_star, compute_angle, compute_curvature, volume_calculator_cyclinder, compute_center_of_catheter, compute_unit_position_vector
import os
import threading
from talk_arduino import arduino_control, distance_arduino
from constants import *
translate =False
def send_arduino_command(command):
    arduino_thread = threading.Thread(target=arduino_control, args=(command,))
    arduino_thread.start()
    return arduino_thread

# -------- functions -------------
def setp_to_list(setp, offset=0):
    return [setp.__dict__[f"input_double_register_{i + offset}"] for i in range(6)]


def list_to_setp(setp, lst, offset=0):
    """ Converts a list into RTDE input registers, allowing different sets with offsets. """
    for i in range(6):
        setp.__dict__[f"input_double_register_{i + offset}"] = lst[i]
    return setp
distance = 40  
travel = str(distance_arduino(distance))
if translate == True:
    arduino_thread = send_arduino_command(f'REV {travel}')
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

for i in range(12):  # Ensure all registers are initialized
    setattr(setp, f"input_double_register_{i}", 0)


setp.input_bit_registers0_to_31 = 0
watchdog.input_int_register_0 = 0

# Start Data Synchronization
if not con.send_start():
    sys.exit()

# Define Waypoints for Continuous Trajectory
waypoints = [
    [0.12054048769760504, 0.4082949683478192, 0.22461119073392152, 0.9719199522201973, 2.9741737060449833, -0.0667004517649879],
    [0.12054246739349583, 0.4783363765516909, 0.22461066969424692, 0.9719123645023171, 2.9741825330052407, -0.06668206717351902] # Start
    # [0.01561346376380156, 0.6392848297605487, 0.3870643751382633, -3.1129835149573597, -0.39044182826995194, -0.02877771799883438],
    # [-0.31561346376380156, 0.6392848297605487, 0.3870643751382633, -3.1129835149573597, -0.39044182826995194, -0.02877771799883438],
    # [-0.31561346376380156, 0.8392848297605487, 0.3870643751382633, -3.1129835149573597, -0.39044182826995194, -0.02877771799883438],
    # [0.01561346376380156, 0.8392848297605487, 0.3870643751382633, -3.1129835149573597, -0.39044182826995194, -0.02877771799883438]    # Start
   # Start
    # [-0.119, 0.616, 0.2, 3.1, -0.5, 0.0],  # Intermediate 2
    # [-0.404, 0.616, 0.2, 3.1, -0.5, 0.0],  # Intermediate 3
    # [-0.404, 0.716, 0.2, 3.1, -0.5, 0.0],  # Intermediate 4
    # [-0.119, 0.716, 0.2, 2.3, -2.05,-0.404, 0.516, 0.2, 0.11, -3.13, -0.04 0.1]   # Final
]


# position_open_loop = [-1.4874456564532679, -1.552175835972168, 2.503701988850729, 3.7785069185444335, -1.5644429365741175, 0.08028531819581985]
position_open_loop2 = [-2.279425923024313, -1.731661935845846, 2.610805336629049, 3.8676439958759765, -1.5944564978228968, 3.068880319595337]
orientation_const = waypoints[1][3:]

trajectory_time_total = 5 # Total time for full motion
trajectory_time_per_segment = trajectory_time_total / (len(waypoints) - 1)  # Split time per segment
state = con.receive()
tcp1 = state.actual_TCP_pose
print(tcp1)
waypoints_list = setp_to_list(setp, offset=0)  # Read waypoints (0-5)
position_list = setp_to_list(setp, offset=6)  # Read position (6-11)

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
list_to_setp(setp, waypoints[0], offset=0)  # Waypoints in registers 0-5
con.send(setp) 

while True:
    print('Waiting for moveJ() first to finish')
    state = con.receive()
    con.send(watchdog)
    if not state.output_bit_registers0_to_31:
        print('Proceeding to mode 2\n')
        break

# Execute ServoJ with Smooth Multi-Waypoint Trajectory
print("-------Executing servoJ  -----------\n")
watchdog.input_int_register_0 = 2
con.send(watchdog)
trajectory_time = 2 # time of min_jerk trajectory
dt = 1/500  # 500 Hz    # frequency
plotter = True
if plotter:
    time_plot = []

    min_jerk_x = []
    min_jerk_y = []
    min_jerk_z = []

    min_jerk_vx = []
    min_jerk_vy = []
    min_jerk_vz = []

    px = []
    py = []
    pz = []

    vx = []
    vy = []
    vz = []


# Multi-Waypoint Execution with Smooth Motion
current_waypoint_index = 0
t_start = time.time()

while current_waypoint_index < len(waypoints) - 1:
    t_segment_start = time.time()
    start_point = waypoints[current_waypoint_index]
    end_point = waypoints[current_waypoint_index + 1]
    planner = PathPlanTranslation(start_point, end_point, trajectory_time_per_segment)

    while time.time() - t_segment_start < trajectory_time_per_segment:
        t_current = time.time() - t_segment_start  # Time for current segment
        state = con.receive()

        if state.runtime_state > 1:
            # Compute smooth trajectory for current segment
            [position_ref, lin_vel_ref, acceleration_ref]= planner.trajectory_planning(t_current)
            current_pose = state.actual_TCP_pose
            current_speed = state.actual_TCP_speed
            pose = position_ref.tolist() + orientation_const
            list_to_setp(setp, pose)
            con.send(setp)

            # Plot data for debugging
            if plotter:
                time_plot.append(time.time() - t_start)

                min_jerk_x.append(position_ref[0])
                min_jerk_y.append(position_ref[1])
                min_jerk_z.append(position_ref[2])

                min_jerk_vx.append(lin_vel_ref[0])
                min_jerk_vy.append(lin_vel_ref[1])
                min_jerk_vz.append(lin_vel_ref[2])

                px.append(current_pose[0])
                py.append(current_pose[1])
                pz.append(current_pose[2])

                vx.append(current_speed[0])
                vy.append(current_speed[1])
                vz.append(current_speed[2])

    current_waypoint_index += 1  # Move to the next waypoint

print(f"It took {time.time()-t_start}s to execute the servoJ")

state = con.receive()
time.sleep(0.5)  # Ensure the robot has time to receive and process the movement

print("-------Executing moveJ -----------\n")
distance = 55
travel = str(distance_arduino(distance))
if translate == True:
    arduino_thread = send_arduino_command(f'REV {travel}')

watchdog.input_int_register_0 = 3
con.send(watchdog)
# Read the actual joint positions before modification
actual_position = state.actual_q
print("Actual joint position before moveJ:", actual_position)

# Compute rotation and modify wrist joint 3 (joint index 5)
rotation_angle_radians = -alpha_star  # Example: adding 100 degrees
position = actual_position[:]  # Copy current joint states
position = [float(joint) for joint in actual_position]  # Convert to float
position[5] += rotation_angle_radians  # Apply rotation

print(f"Sent joint positions: {position}")
list_to_setp(setp, position, offset=6)
con.send(setp)
time.sleep(0.5)  # Allow time for movement

# Print what will be sent


# Send Initial Pose

# list_to_setp(setp, position, offset=6)  # Store position in registers 6-11
# con.send(setp)

while True:
    print('Waiting for moveJ() second to finish...')
    state = con.receive()
    con.send(watchdog)
    if not state.output_bit_registers0_to_31:
        print('MoveJ completed, proceeding to next mode\n')
        break

# Print actual joint positions after execution
state = con.receive()
print('--------------------')
print("Actual joint position after moveJ:", state.actual_q)
print("Actual TCP pose after moveJ:", state.actual_TCP_pose)
# Mode 3


time.sleep(2)
print("-------Executing servoJ  -----------\n")
watchdog.input_int_register_0 = 4
con.send(watchdog)
trajectory_time = 3  # time of min_jerk trajectory
dt = 1/500  # 500 Hz    # frequency
plotter = True
if plotter:
    time_plot = []

    min_jerk_x = []
    min_jerk_y = []
    min_jerk_z = []

    min_jerk_vx = []
    min_jerk_vy = []
    min_jerk_vz = []

    px = []
    py = []
    pz = []

    vx = []
    vy = []
    vz = []


# Multi-Waypoint Execution with Smooth Motion
current_waypoint_index2 = 0
t_start = time.time()
actual_position2 = state.actual_TCP_pose

position2 = [float(joint) for joint in actual_position2] 
waypoints2 = [
    position2, # Start
    [0.12054659833862652, 0.5287865839067847, 0.22462264096441512, 0.9718915055124021, 2.974158985078232, -0.06666407249521322]
]
orientation_const2 = waypoints2[0][3:]

while current_waypoint_index2 < len(waypoints2) - 1:
    t_segment_start = time.time()
    start_point2 = waypoints2[current_waypoint_index2]
    end_point2 = waypoints2[current_waypoint_index2 + 1]
    planner = PathPlanTranslation(start_point2, end_point2, trajectory_time_per_segment)

    while time.time() - t_segment_start < trajectory_time_per_segment:
        t_current = time.time() - t_segment_start  # Time for current segment
        state = con.receive()

        if state.runtime_state > 1:
            # Compute smooth trajectory for current segment
            [position_ref, lin_vel_ref, acceleration_ref]= planner.trajectory_planning(t_current)
            current_pose = state.actual_TCP_pose
            current_speed = state.actual_TCP_speed
            pose = position_ref.tolist() + orientation_const2
            list_to_setp(setp, pose)
            con.send(setp)

            # Plot data for debugging
            if plotter:
                time_plot.append(time.time() - t_start)

                min_jerk_x.append(position_ref[0])
                min_jerk_y.append(position_ref[1])
                min_jerk_z.append(position_ref[2])

                min_jerk_vx.append(lin_vel_ref[0])
                min_jerk_vy.append(lin_vel_ref[1])
                min_jerk_vz.append(lin_vel_ref[2])

                px.append(current_pose[0])
                py.append(current_pose[1])
                pz.append(current_pose[2])

                vx.append(current_speed[0])
                vy.append(current_speed[1])
                vz.append(current_speed[2])

    current_waypoint_index2 += 1  # Move to the next waypoint

print(f"It took {time.time()-t_start}s to execute the servoJ")

state = con.receive()
time.sleep(0.5)  # Ensure the robot has time to receive and process the movement

print("-------Executing moveJ -----------\n")
kappa = compute_curvature(theta_l, length_c)
volume_cath = volume_calculator_cyclinder((s_c/2), length_c_m)
volume_mag = volume_calculator_cyclinder((h_a/2), d_a)
x_c, y_c = compute_center_of_catheter(length_c_m, kappa, theta_l)
p_hat, p_norm1, y_p, x_p = compute_unit_position_vector(x_c, y_c, d, h)
alpha_star2 = compute_angle(EI, 30, length_c, x_p, y_p, x_basis, y_basis, p_norm1)
watchdog.input_int_register_0 = 5
# Read the actual joint positions before modification
actual_position = state.actual_q
print("Actual joint position before moveJ:", actual_position)

# Compute rotation and modify wrist joint 3 (joint index 5)
rotation_angle_radians = -alpha_star2 # Example: adding 100 degrees
position2 = actual_position[:]  # Copy current joint states
position2 = [float(joint) for joint in actual_position]  # Convert to float
position2[5] += rotation_angle_radians  # Apply rotation

print(f"Sent joint positions: {position}")
list_to_setp(setp, position2, offset=6)
con.send(setp)
time.sleep(0.5)  # Allow time for movement

# Print what will be sent


# Send Initial Pose

con.send(watchdog)
# list_to_setp(setp, position, offset=6)  # Store position in registers 6-11
# con.send(setp)

while True:
    print('Waiting for moveJ() second to finish...')
    state = con.receive()
    con.send(watchdog)
    if not state.output_bit_registers0_to_31:
        print('MoveJ completed, proceeding to next mode\n')
        break

# Print actual joint positions after execution
state = con.receive()
print('--------------------')
print("Actual joint position after moveJ:", state.actual_q)
print("Actual TCP pose after moveJ:", state.actual_TCP_pose)



watchdog.input_int_register_0 = 6
con.send(watchdog)

con.send_pause()
con.disconnect()
distance = 25
travel = str(distance_arduino(distance))
if translate == True:
    arduino_thread = send_arduino_command(f'REV {travel}')
# Plot Results
# Plot Results
if plotter:
    # ----------- Position Plots -------------
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    axs[0].plot(time_plot, min_jerk_x, label="x_min_jerk")
    axs[0].plot(time_plot, px, label="x_robot")
    axs[0].set_ylabel('Position in x [m]')
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(time_plot, min_jerk_y, label="y_min_jerk")
    axs[1].plot(time_plot, py, label="y_robot")
    axs[1].set_ylabel('Position in y [m]')
    axs[1].legend()
    axs[1].grid()

    axs[2].plot(time_plot, min_jerk_z, label="z_min_jerk")
    axs[2].plot(time_plot, pz, label="z_robot")
    axs[2].set_ylabel('Position in z [m]')
    axs[2].set_xlabel('Time [sec]')
    axs[2].legend()
    axs[2].grid()

    plt.show()

    # ----------- Velocity Plots -------------
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    axs[0].plot(time_plot, min_jerk_vx, label="vx_min_jerk")
    axs[0].plot(time_plot, vx, label="vx_robot")
    axs[0].set_ylabel('Velocity in x [m/s]')
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(time_plot, min_jerk_vy, label="vy_min_jerk")
    axs[1].plot(time_plot, vy, label="vy_robot")
    axs[1].set_ylabel('Velocity in y [m/s]')
    axs[1].legend()
    axs[1].grid()

    axs[2].plot(time_plot, min_jerk_vz, label="vz_min_jerk")
    axs[2].plot(time_plot, vz, label="vz_robot")
    axs[2].set_ylabel('Velocity in z [m/s]')
    axs[2].set_xlabel('Time [sec]')
    axs[2].legend()
    axs[2].grid()

    plt.show()