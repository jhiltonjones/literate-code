#inv_joint_work_w_relative.urp, this uses PID control from the inputed spline to move the robot arm in relation to the catheter. The translation is constant speed.

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
from camera_to_robot_frame import pixel_to_robot_frame
from PID_control import PIDController
from new_cam import new_capture
from new_tip_finder import detect_rod_tip_darkest_right
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
translate =False
max_attempts =0
pid = PIDController(Kp=0.05, Ki=0.005, Kd=0.005, dt=0.1)

def send_arduino_command(command):
    arduino_thread = threading.Thread(target=arduino_control, args=(command,))
    arduino_thread.start()
    return arduino_thread

distance = 165
travel = str(distance_arduino(distance))
if translate == True:
    arduino_thread = send_arduino_command(f'ON {travel}')

def get_inverse(con, setp, position_inv):
    # Set registers
    list_to_setp(setp, position_inv, offset=12)
    con.send(setp)

    time.sleep(0.1)

    # Read back the joint values
    state = con.receive()
    joints = [
        getattr(state, 'output_double_register_12'),
        getattr(state, 'output_double_register_13'),
        getattr(state, 'output_double_register_14'),
        getattr(state, 'output_double_register_15'),
        getattr(state, 'output_double_register_16'),
        getattr(state, 'output_double_register_17'),
    ]
    return joints


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


setp.input_double_register_12 = 0
setp.input_double_register_13 = 0
setp.input_double_register_14 = 0
setp.input_double_register_15 = 0
setp.input_double_register_16 = 0
setp.input_double_register_17 = 0
setp.input_bit_registers0_to_31 = 0
# watchdog.input_int_register_0 = 0

# start data synchronization
if not con.send_start():
    sys.exit()


transformed_points = transform_point(T, d, h)
x_robotic_arm = transformed_points[0]
y_robotic_arm = transformed_points[1]

start_point = [0.3155444774985512, 0.33140389190909014, 0.279839378899992, 0.36777814360853694, 3.089904007729656, 0.021724405192601978]
start_point2 = [-2.7212381998645228, -1.765574117700094, 2.5149717966662806, 3.9297119814106445, -1.5702789465533655, -4.051521603261129]

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
print("-------Executing moveJ start -----------\n")

print("-------Executing moveJ -----------\n")

watchdog.input_int_register_0 = 1
con.send(watchdog)
# list_to_setp(setp, start_point, offset=12)
list_to_setp(setp, start_point, offset=12)
list_to_setp(setp, start_point2, offset=6)

con.send(setp)
time.sleep(0.5)

# list_to_setp(setp, start_point2, offset=6)
# con.send(setp)


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


# Initial rotation offset in joint space
joint6_angle = None
rotation_step = 0.2
attempts = 0


while attempts < max_attempts:
    try:
        state = con.receive()
        if state is None:
            print("Disconnected from robot.")
            break
        con.send(watchdog)
        # image_path = capture_image()
        image_path = new_capture()
        tip, rod_pos, error, desired_point = detect_rod_tip_darkest_right(image_path, graph=False)
        # tip, rod_pos, error, desired_point = below_or_above(image_path, False)
        robotposx, robotposy = pixel_to_robot_frame(rod_pos[0], rod_pos[1])
        robotposy -= 0.1
        robotposy = np.clip(robotposy, 0.15541394046736406, 0.367075638609634)
        # Keep a fixed pose orientation
        position = [robotposx, robotposy, 0.279839378899992, 0.36777814360853694, 3.089904007729656, 0.021724405192601978]

        list_to_setp(setp, position, offset=12)
        joints_off = get_inverse(con, setp, position)

        if joint6_angle is None:
            # Only set once from first IK
            joint6_angle = joints_off[5]



        rotation_adjustment = pid.update(error)
        joint6_angle -= (rotation_adjustment)
        joint6_angle = np.clip(joint6_angle, -6.115246836339132, -2.434729878102438)
        print(f"PID Rotation Adjustment: {rotation_adjustment}")
        # if tip == "Below":
        #     joint6_angle -= rotation_step
        #     print("Below")
        # elif tip == "Above":
        #     joint6_angle += rotation_step
        #     print("Above")
        # else:
        #     print("Rod tip aligned with spline. No further adjustment.")
        #     break

        # Apply modified joint6
        joints_off[5] = joint6_angle

        print(f"Moving magnet to position (joints): {joints_off}")
        list_to_setp(setp, joints_off, offset=6)
        con.send(setp)
        time.sleep(0.2)
        # con.send(watchdog)

        while True:
            state = con.receive()
            if state is None:
                print("Lost connection while waiting.")
                break
            con.send(watchdog)
            if not state.output_bit_registers0_to_31:
                break
    except Exception as e:
        print(f"Error occurred: {e}")
        break

    attempts += 1



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
# image_final = capture_image()
time.sleep(3)
travel = str(distance_arduino(0))
if translate == True:
    arduino_thread = send_arduino_command(f'ON {travel}')