import sys
import logging
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config
import time
import numpy as np
import matplotlib.pyplot as plt
from new_cam import new_capture
import pandas as pd
import os
import cv2
def setp_to_list(setp, offset=0):
    return [setp.__dict__[f"input_double_register_{i + offset}"] for i in range(6)]


def list_to_setp(setp, lst, offset=0):
    """ Converts a list into RTDE input registers, allowing different sets with offsets. """
    for i in range(6):
        setp.__dict__[f"input_double_register_{i + offset}"] = lst[i]
    return setp
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

# Clear bit registersFalse
setp.input_bit_registers0_to_31 = 0
watchdog.input_int_register_0 = 0

setp.input_double_register_16 = 0
setp.input_double_register_17 = 0
setp.input_bit_registers0_to_31 = 0


if not con.send_start():
    sys.exit()
start_point = [0.6863228126720625, -0.49297247483218243, 0.39899908106866633, 2.3176671913808944, 2.117819880135014, -0.057175881373984896]#tcp 0cm 16cm in y
start_point2= [-0.4110644499408167, -1.8883592091002406, -1.699528455734253, -1.1374167960933228, 1.5391621589660645, -3.4619577566729944]


start_point_list = setp_to_list(setp, offset=0)
position_list = setp_to_list(setp, offset=6)  
inverse_list = setp_to_list(setp, offset=12)
reset = True


while True:
    print('Boolean 1 is False, please click CONTINUE on the Polyscope')
    state = con.receive()
    con.send(watchdog)
    if state.output_bit_registers0_to_31:  
        print('Boolean 1 is True, proceeding to mode 1\n')
        break

print("-------Executing moveJ -----------\n")

watchdog.input_int_register_0 = 1
con.send(watchdog)
state = con.receive()
joint_state_start = state.actual_q
start_point2[5] = start_point2[5] - np.pi/2
# start_point2[5] = start_point3[5] 

# list_to_setp(setp, start_point, offset=12)
list_to_setp(setp, joint_state_start, offset=6)

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

# Read and store the initial joint state once before the loop
state = con.receive()
if state is None:
    print("Disconnected from robot.")
    sys.exit()
initial_joints = list(state.actual_q)

attempts = 0
max_attempts = 150
increment = 0.03  # Radians per attempt
# max_angle = start_point2[5] + np.pi/2
max_angle = joint_state_start[5] + np.pi/2
while attempts < max_attempts:
    try:


    attempts += 1

watchdog.input_int_register_0 = 3
con.send(watchdog) 
state = con.receive()
terminal_joint =state.actual_q
print(f"terminal joint state is {terminal_joint[5]}")
con.send_pause()
con.disconnect()
print("Disconnected")


