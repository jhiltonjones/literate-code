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
from constants import d, h, theta_l, EI, x_basis, y_basis
from PID_control import PIDController
from image_capture import capture_image
from bending_calculation import calculate_bending_angle
import os
from tip_w_spline import below_or_above
import threading
from talk_arduino import arduino_control, distance_arduino
plot = False
# time.sleep(3)
# log_dir = "/home/jack/literate-code/"
# os.makedirs(log_dir, exist_ok=True)
# log_file_path = os.path.join(log_dir, "steering_angles_log.txt")

# def log_steering_angles(desired_angle, actual_angle, filename=log_file_path):
#     """Logs the desired and actual steering angles to a text file."""
#     log_entry = f"Desired Angle: {desired_angle:.2f} degrees, Actual Angle: {actual_angle:.2f} degrees\n"

#     with open(filename, "a") as file:
#         file.write(log_entry)

#     print(f"Logged: {log_entry.strip()}")
translate =True
def send_arduino_command(command):
    arduino_thread = threading.Thread(target=arduino_control, args=(command,))
    arduino_thread.start()
    return arduino_thread

distance = 150
travel = str(distance_arduino(distance))
if translate == True:
    arduino_thread = send_arduino_command(f'REV {travel}')

def setp_to_list(setp, offset=0):
    return [setp.__dict__[f"input_double_register_{i + offset}"] for i in range(6)]


def list_to_setp(setp, lst, offset=0):
    """ Converts a list into RTDE input registers, allowing different sets with offsets. """
    for i in range(6):
        setp.__dict__[f"input_double_register_{i + offset}"] = lst[i]
    return setp
time.sleep(2)
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

# Clear all input double registers
for i in range(24):  # Go beyond 12 if your config might use more
    setattr(setp, f"input_double_register_{i}", 0.0)

# Clear bit registers
setp.input_bit_registers0_to_31 = 0
watchdog.input_int_register_0 = 0

# Send zeroed setup to robot
con.send(setp)
con.send(watchdog)


# setp.input_double_register_0 = 0
# setp.input_double_register_1 = 0
# setp.input_double_register_2 = 0
# setp.input_double_register_3 = 0
# setp.input_double_register_4 = 0
# setp.input_double_register_5 = 0
# setp.input_bit_registers0_to_31 = 0
# watchdog.input_int_register_0 = 0

# start data synchronization
if not con.send_start():
    sys.exit()


transformed_points = transform_point(T, d, h)
x_robotic_arm = transformed_points[0]
y_robotic_arm = transformed_points[1]

start_point = [-2.2383063475238245, -1.846382280389303, 2.72556716600527, 3.8191911417194824, -1.6096790472613733, 0.7579470872879028]

waypoints = [
    [-2.2383063475238245, -1.846382280389303, 2.72556716600527, 3.8191911417194824, -1.6096790472613733, 0.7579470872879028],
    [-2.1899974981891077, -1.7748571834959925, 2.6824050585376185, 3.792893095607422, -1.6103337446795862, 1.0392372608184814],
    [-2.167614761983053, -1.7413836918272914, 2.659560743962423, 3.783254309291504, -1.6106246153460901, 1.0617380142211914], # Start
    [-2.1405447165118616, -1.7004362545409144, 2.62907845178713, 3.773951216334961, -1.610938851033346, 1.0888798236846924],
    [-2.1037171522723597, -1.6432873211302699, 2.581790272389547, 3.7657391267963867, -1.611258331929342, 1.1258397102355957],
    [-2.065061394368307, -1.5807472668089808, 2.523339811955587, 3.7633725839802246, -1.6115606466876429, 1.8864836692810059], # Start
    [-2.0577953497516077, -1.5686908706887444, 2.5112608114825647, 3.7637011247822265, -1.6115935484515589, 1.8937296867370605]
    # [-0.31561346376380156, 0.8392848297605487, 0.3870643751382633, -3.1129835149573597, -0.39044182826995194, -0.02877771799883438],
    # [-0.31561346376380156, 0.4392848297605487, 0.3870643751382633, -3.1129835149573597, -0.39044182826995194, -0.02877771799883438]
    # [-0.31561346376380156, 0.8392848297605487, 0.3870643751382633, -3.1129835149573597, -0.39044182826995194, -0.02877771799883438],
    # [0.01561346376380156, 0.8392848297605487, 0.3870643751382633, -3.1129835149573597, -0.39044182826995194, -0.02877771799883438] 
]

start_point_list = setp_to_list(setp, offset=6)
position_list = setp_to_list(setp, offset=6)  
reset = True

while True:
    print('Boolean 1 is False, please click CONTINUE on the Polyscope')
    state = con.receive()
    con.send(watchdog)
    if state.output_bit_registers0_to_31:  
        print('Boolean 1 is True, proceeding to mode 1\n')
        break
print("-------Executing moveJ start -----------\n")

max_attempts = 9


print("-------Executing moveJ -----------\n")

watchdog.input_int_register_0 = 1
con.send(watchdog)
list_to_setp(setp, start_point, offset=6)
con.send(setp)
time.sleep(0.5)

while True:
    print(f'Waiting for moveJ() to finish start...')
    state = con.receive()
    con.send(watchdog)
    if not state.output_bit_registers0_to_31:
        print('Start completed, proceeding to feedback check\n')
        break

watchdog.input_int_register_0 = 2
con.send(watchdog)
state = con.receive()
actual_position = state.actual_q
position = [float(joint) for joint in actual_position] 
image_path = capture_image()
tip = below_or_above(image_path)
rotation_step = 0.35
adjustment = True
attempts = 0
i = 0
while adjustment and attempts < max_attempts:
    # time.sleep(1)
    state = con.receive()
    actual_position = state.actual_q
    position = [float(joint) for joint in actual_position]

    image_path = capture_image()
    tip = below_or_above(image_path)
    if tip == "Below":
        position[5] -= rotation_step
        
    elif tip == "Above":
        position[5] += rotation_step
    else:
        print("Rod tip aligned with spline. No further adjustment.")
        adjustment = False
    attempts += 1
    position_next = waypoints[i]
    position_next[5] = position[5]


    print(f"Moving magnet to position: {position}")
    list_to_setp(setp, position_next, offset=6)
    con.send(setp)
    time.sleep(0.5) 
    con.send(watchdog)

    while True:
        print(f'Waiting for moveJ() to finish...')
        state = con.receive()
        con.send(watchdog)
        if not state.output_bit_registers0_to_31:
            print('MoveJ completed, proceeding to feedback check\n')
            break
    if i < 4:
        i +=1

watchdog.input_int_register_0 = 3
con.send(watchdog) 
time.sleep(1.5) 
# Optionally clear registers before shutdown
# for i in range(24):
#     setattr(setp, f"input_double_register_{i}", 0.0)
# setp.input_bit_registers0_to_31 = 0
# watchdog.input_int_register_0 = 0
# con.send(setp)
# con.send(watchdog)

con.send_pause()
con.disconnect()
print("Disconnected")
image_final = capture_image()
below_or_above(image_final)
time.sleep(3)
below_or_above(image_final)
travel = str(distance_arduino(0))
if translate == True:
    arduino_thread = send_arduino_command(f'ON {travel}')