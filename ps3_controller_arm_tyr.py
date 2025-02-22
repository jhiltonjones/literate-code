import sys
import logging
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config
import time
import evdev
import numpy as np
import matplotlib.pyplot as plt
from trajectory_planner_line import PathPlanTranslation 
from rotation_matrix import T, transform_point
from bfgs_minimise import alpha_star, alpha_star_deg, compute_angle, length_c, x_p, y_p, p_norm1
from constants import d, h, theta_l, EI, x_basis, y_basis
from PID_control import PIDController
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


transformed_points = transform_point(T, d, h)
x_robotic_arm = transformed_points[0]
y_robotic_arm = transformed_points[1]

# start_point = [-1.7749140898333948, -1.8188401661314906, 2.519264046345846, 4.06753317892041, -1.5926616827594202, 0.3149070739746094]
start_point = [-1.7749140898333948, -1.8188401661314906, 2.519264046345846, 4.06753317892041, -1.5926616827594202, 0.3149070739746094]

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


max_attempts = 3
# rotation_step = 0.05 
vessel_branch_target_angle = 45 

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


state = con.receive()
actual_position = state.actual_q
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
device_path = "/dev/input/event262"  
try:
    controller = evdev.InputDevice(device_path)
    print(f"Using device: {controller.path} ({controller.name})")
except FileNotFoundError:
    print(f"Device not found at {device_path}. Check permissions and connection.")
    sys.exit()

# --------------------- Robot Control Variables ---------------------
joystick_deadzone = 10  
scale_factor = 0.01  

# --------------------- Read Controller Input and Move Robot ---------------------
print("Use the PS3 controller to move the UR robot. Press Ctrl+C to exit.")

move_x = 0
move_y = 0
move_rotation = 0

try:
    for event in controller.read_loop():
        if event.type == evdev.ecodes.EV_ABS:
            event_code = evdev.ecodes.bytype[event.type].get(event.code, f"Unknown({event.code})")
            event_value = event.value

            if event_code == "ABS_X":  
                move_x = (event_value - 128) / 128.0
                if abs(move_x) > (joystick_deadzone / 128.0):
                    move_x *= scale_factor
                else:
                    move_x = 0

            elif event_code == "ABS_Y":  
                move_y = (event_value - 128) / 128.0
                if abs(move_y) > (joystick_deadzone / 128.0):
                    move_y *= scale_factor
                else:
                    move_y = 0

            elif event_code == "ABS_RY":  
                move_rotation = (event_value - 128) / 128.0
                if abs(move_rotation) > (joystick_deadzone / 128.0):
                    move_rotation *= scale_factor
                else:
                    move_rotation = 0

        # ---------------- Update Robot Position ---------------------
        state = con.receive()
        actual_position = list(state.actual_q)

        actual_position[0] += move_x
        actual_position[1] += move_y
        actual_position[5] += move_rotation

        for i in range(6):
            setp.__dict__[f"input_double_register_{i + 6}"] = actual_position[i]

        if con.is_connected():
            try:
                con.send(setp)
            except BrokenPipeError:
                print("⚠️ Connection lost! Reconnecting...")
                con.disconnect()
                time.sleep(1)
                con.connect()
                con.send(setp)  

        con.send(watchdog)
        time.sleep(0.01)  

except KeyboardInterrupt:
    print("\nStopping robot and exiting.")
    watchdog.input_int_register_0 = 4
    con.send(watchdog)
    con.send_pause()
    con.disconnect()

watchdog.input_int_register_0 = 4
con.send(watchdog)
time.sleep(1.5) 
con.send_pause()
con.disconnect()
print("Disconnected")
