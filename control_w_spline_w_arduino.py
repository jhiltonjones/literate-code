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

distance = 90  
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

start_point = [-2.1419737974749964, -1.8592311344542445, 2.648673121129171, 3.8949782091328125, -1.612434212361471, 2.9840893745422363]

waypoints = [
    [-2.1115439573871058, -1.8060242138304652, 2.61625844637026, 3.8756539064594726, -1.6132920424090784, 3.0146420001983643],
    [-2.098027054463522, -1.781929155389303, 2.600307289754049, 3.86812607824292, -1.6136663595782679, 3.0282578468322754],
    [-2.085815731679098, -1.7599970302977503, 2.58505088487734, 3.862001581782959, -1.6139710585223597, 3.040501117706299], # Start
    [-2.095215622578756, -1.7362853489317835, 2.5681217352496546, 3.8547751146503906, -1.6137741247760218, 3.031151056289673],
    [-2.104853932057516, -1.7101088962950648, 2.5484705607043665, 3.8478452402302246, -1.6135438124286097, 3.021568775177002]
    # [0.11561346376380156, 0.4392848297605487, 0.3870643751382633, -3.1129835149573597, -0.39044182826995194, -0.02877771799883438], # Start
    # [0.01561346376380156, 0.8392848297605487, 0.3870643751382633, -3.1129835149573597, -0.39044182826995194, -0.02877771799883438],
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

max_attempts = 7


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
rotation_step = 0.5
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
for i in range(24):
    setattr(setp, f"input_double_register_{i}", 0.0)
setp.input_bit_registers0_to_31 = 0
watchdog.input_int_register_0 = 0
con.send(setp)
con.send(watchdog)

con.send_pause()
con.disconnect()
print("Disconnected")
image_final = capture_image()
below_or_above(image_final)

travel = str(distance_arduino(distance-15))
if translate == True:
    arduino_thread = send_arduino_command(f'ON {travel}')