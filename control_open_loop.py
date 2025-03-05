import sys
import logging
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config
import time
import numpy as np
import matplotlib.pyplot as plt
from trajectory_planner_line import PathPlanTranslation 
from rotation_matrix import T, transform_point
from bfgs_minimise import alpha_star, alpha_star_deg, compute_angle, length_c, x_p, y_p, p_norm1
from constants import d, h, theta_l, theta_l2, EI, x_basis, y_basis
from PID_control import PIDController
from image_capture import capture_image
from bending_calculation import calculate_bending_angle
import os
import threading
from talk_arduino import arduino_control, distance_arduino

finished = False
translate = True
plot = True


def send_arduino_command(command):
    arduino_thread = threading.Thread(target=arduino_control, args=(command,))
    arduino_thread.start()
    return arduino_thread
log_dir = "/home/jack/literate-code/"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "steering_angles_log.txt")

def log_steering_angles(desired_angle, actual_angle, filename=log_file_path):
    """Logs the desired and actual steering angles to a text file."""
    log_entry = f"Desired Angle: {desired_angle:.2f} degrees, Actual Angle: {actual_angle:.2f} degrees\n"

    with open(filename, "a") as file:
        file.write(log_entry)

    print(f"Logged: {log_entry.strip()}")




def setp_to_list(setp, offset=0):
    return [setp.__dict__[f"input_double_register_{i + offset}"] for i in range(6)]


def list_to_setp(setp, lst, offset=0):
    """ Converts a list into RTDE input registers, allowing different sets with offsets. """
    for i in range(6):
        setp.__dict__[f"input_double_register_{i + offset}"] = lst[i]
    return setp

distance = 80
travel = str(distance_arduino(distance))
if translate == True:
    # while finished is not True:
    arduino_thread = send_arduino_command(f'REV {travel}')
    finished = True
finished = False

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


transformed_points = transform_point(T, d, h)
x_robotic_arm = transformed_points[0]
y_robotic_arm = transformed_points[1]

start_point = [-1.5734875837909144, -1.3800104421428223, 2.4826610724078577, 3.6008288103291015, -1.5717714468585413, 0.5280584692955017]

waypoints = [
    [-2.7217212359057825, -1.2940319341472168, 1.286574665700094, -1.5805064640440882, -1.5797279516803187, 2.6362221240997314],
    [x_robotic_arm, y_robotic_arm, 0.1870643751382633, -3.1129835149573597, -0.39044182826995194, -0.02877771799883438],
    # [0.11561346376380156, 0.4392848297605487, 0.3870643751382633, -3.1129835149573597, -0.39044182826995194, -0.02877771799883438], # Start
    # [0.01561346376380156, 0.8392848297605487, 0.3870643751382633, -3.1129835149573597, -0.39044182826995194, -0.02877771799883438],
    # [-0.31561346376380156, 0.8392848297605487, 0.3870643751382633, -3.1129835149573597, -0.39044182826995194, -0.02877771799883438],
    # [-0.31561346376380156, 0.4392848297605487, 0.3870643751382633, -3.1129835149573597, -0.39044182826995194, -0.02877771799883438],
    # [0.11561346376380156, 0.4392848297605487, 0.3870643751382633, -3.1129835149573597, -0.39044182826995194, -0.02877771799883438], # Start
    # [0.01561346376380156, 0.8392848297605487, 0.3870643751382633, -3.1129835149573597, -0.39044182826995194, -0.02877771799883438],
    # [-0.31561346376380156, 0.8392848297605487, 0.3870643751382633, -3.1129835149573597, -0.39044182826995194, -0.02877771799883438],
    # [-0.31561346376380156, 0.4392848297605487, 0.3870643751382633, -3.1129835149573597, -0.39044182826995194, -0.02877771799883438]
    # [-0.31561346376380156, 0.8392848297605487, 0.3870643751382633, -3.1129835149573597, -0.39044182826995194, -0.02877771799883438],
    # [0.01561346376380156, 0.8392848297605487, 0.3870643751382633, -3.1129835149573597, -0.39044182826995194, -0.02877771799883438] 
]

orientation_const = waypoints[0][3:]
trajectory_time_total = 5
trajectory_time_per_segment = trajectory_time_total / (len(waypoints) - 1) 
state = con.receive()
tcp1 = state.actual_TCP_pose
print(tcp1)
start_point_list = setp_to_list(setp, offset=6)
waypoints_list = setp_to_list(setp, offset=0) 
position_list = setp_to_list(setp, offset=6)  
reset = True

while True:
    # print('Boolean 1 is False, please click CONTINUE on the Polyscope')
    state = con.receive()
    con.send(watchdog)
    if state.output_bit_registers0_to_31:  
        print('Boolean 1 is True, proceeding to mode 1\n')
        break
print("-------Executing moveJ start -----------\n")


pid = PIDController(Kp=0.5, Ki=0.1, Kd=0.05, dt=0.1)


max_attempts = 6
demos = 2
j = 0
# rotation_step = 0.05 
vessel_branch_target_angle = theta_l

print("-------Executing moveJ with PID -----------\n")

watchdog.input_int_register_0 = 1
con.send(watchdog)
list_to_setp(setp, start_point, offset=6)
con.send(setp)
time.sleep(0.5)

while True:
    # print(f'Waiting for moveJ() to finish... {i}')
    state = con.receive()
    con.send(watchdog)
    if not state.output_bit_registers0_to_31:
        print('Start completed, proceeding to feedback check\n')
        break
time.sleep(1.5)
watchdog.input_int_register_0 = 2
con.send(watchdog)
state = con.receive()
actual_position = state.actual_q
alpha_star = compute_angle(EI, theta_l, length_c, x_p, y_p, x_basis, y_basis, p_norm1)
alpha_star_deg = np.rad2deg(alpha_star)
print(f"Initial move: Applying rotation angle of {alpha_star}")
position = [float(joint) for joint in actual_position] 
position[5] += alpha_star 
print(f"Moving magnet to position: {position}")
list_to_setp(setp, position, offset=6)
con.send(setp)
time.sleep(0.5) 
con.send(watchdog)

while True:
    # print(f'Waiting for moveJ() to finish... {i}')
    state = con.receive()
    con.send(watchdog)
    if not state.output_bit_registers0_to_31:
        print('MoveJ completed, proceeding to feedback check\n')
        break

print("Final movement completed.")
print('--------------------')
print("Actual joint position after moveJ:", state.actual_q)
print("Actual TCP pose after moveJ:", state.actual_TCP_pose)
print("Moved to position: ", d,h)
print("With a rotation of: ", alpha_star_deg)
print("With a steering angle of: ", np.rad2deg(theta_l))


watchdog.input_int_register_0 = 3
con.send(watchdog)
time.sleep(1.5) 
con.send_pause()
con.disconnect()
print("Disconnected")
time.sleep(8)
if translate is True:
    while finished is not True:
        travel = str(distance_arduino(distance))
        arduino_thread = send_arduino_command(f'ON {travel}')
        finished = True