

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
log=True
def compute_signed_angle(v1, v2):
    """Returns the signed angle in degrees from v1 to v2 (positive = CCW, negative = CW)"""
    angle1 = np.arctan2(v1[1], v1[0])
    angle2 = np.arctan2(v2[1], v2[0])
    angle_rad = angle2 - angle1
    angle_deg = np.degrees(angle_rad)

    # Normalize to [-180, 180]
    if angle_deg > 180:
        angle_deg -= 360
    elif angle_deg < -180:
        angle_deg += 360

    return angle_deg

def detect_red_points_and_angle(image_path, show=False):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    red_ranges = [
        (np.array([0, 50, 50]), np.array([10, 255, 255])),
        (np.array([160, 50, 50]), np.array([180, 255, 255]))
    ]

    red_mask = None
    for lower_red, upper_red in red_ranges:
        temp_mask = cv2.inRange(image_hsv, lower_red, upper_red)
        red_mask = temp_mask if red_mask is None else cv2.bitwise_or(red_mask, temp_mask)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 2:
        raise ValueError("Less than two red points detected!")

    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    red_centers = []
    for cnt in sorted_contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            red_centers.append((cx, cy))
            cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)

    pt1, pt2 = red_centers
    vector = np.array(pt2) - np.array(pt1)
    reference = np.array([1, 0])  # x-axis

    angle = compute_signed_angle(reference, vector)

    if show:
        cv2.line(image, pt1, pt2, (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f"Beam Angle: {angle:.2f}°")
        plt.axis("off")
        plt.show()

    return pt1, pt2, angle
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

# Clear bit registersFalse
setp.input_bit_registers0_to_31 = 0
watchdog.input_int_register_0 = 0

# Send zeroed setup to robot28 = 0
setp.input_double_register_16 = 0
setp.input_double_register_17 = 0
setp.input_bit_registers0_to_31 = 0
# watchdog.input_int_register_0 = 0

# start data synchronization
if not con.send_start():
    sys.exit()

# start_point = [0.5736422825900863, -0.7120664706491283, 0.11347020378486845, -1.713054215587369, -2.603307601635303, -0.0452342048711784]#tcp 14.5cm
# start_point2 = [-0.7047150770770472, -2.265031477014059, -1.699472188949585, -0.7755890649608155, 1.59804368019104, 2.030656099319458]#joints
# start_point = [0.5536482874718568, -0.712126528996371, 0.11351489194233098, -1.7133019013974415, -2.6032316963415454, -0.04512618461746381]#tcp 16.5cm
# start_point2 = [-0.7193802038775843, -2.250943800012106, -1.7275197505950928, -0.7619208854487916, 1.5975193977355957, 2.0162007808685303]#joints
# start_point = [0.5336565464788305, -0.7121068155682757, 0.11351069520350286, -1.7132216476947364, -2.603281876341249, -0.04514274641932209]#tcp 18.5cm
# start_point2 = [-0.7344430128680628, -2.237301012078756, -1.7547472715377808, -0.748717264538147, 1.5970474481582642, 2.0011239051818848]#joints
# start_point = [0.5136455487184496, -0.7120797067295713, 0.11348058291800132, -1.7131391716644324, -2.6032608976902516, -0.04518225339617722]#tcp 20.5cm
# start_point2 = [-0.7499859968768519, -2.224119802514547, -1.7810719013214111, -0.7360222500613709, 1.5965626239776611, 1.9855883121490479]#joints
# start_point = [0.49364556687645744, -0.7120635482456205, 0.11347880608944771, -1.7130855796489652, -2.603297956756908, -0.045211109940160435]#tcp 22.5cm
# start_point2 = [-0.7660134474383753, -2.211367269555563, -1.8064552545547485, -0.7237804693034668, 1.5960613489151, 1.9695625305175781]#joints


# start_point = [0.5536488450634438, -0.7320652960373009, 0.1135122420106864, -1.71332968251415, -2.6031873873441627, -0.045181316562570746]#tcp 16.5cm 2cm in y
# start_point2 = [-0.7359426657306116, -2.2689033947386683, -1.6913682222366333, -0.780555323963501, 1.5971418619155884, 1.999606728553772]
# start_point = [0.5536390628501511, -0.7520670276634125, 0.11350151688715632, -1.7133290981850664, -2.6032191894953556, -0.04515480646310208]#tcp 16.5cm 4cm in y
# start_point2= [-0.7520225683795374, -2.28737272838735, -1.6542009115219116, -0.7996226114085694, 1.5967273712158203, 1.983452320098877]
# start_point = [0.5536689365876796, -0.7719998020450638, 0.11352761424037995, -1.7133363293553172, -2.6032770389949156, -0.045185167101490266]#tcp 16.5cm 4cm in y
# start_point2= [-0.7675078550921839, -2.3062344990172328, -1.61613130569458, -0.8191578549197693, 1.5963668823242188, 1.9678850173950195]
# start_point = [0.5536413692436498, -0.7919986339951873, 0.11350397626943606, -0.6462752977707401, -3.0610844796746286, -0.056420074133725844]#tcp 16.5cm 6cm in y
# start_point2= [-0.7825849691974085, -2.325672765771383, -1.5769622325897217, -0.8393123906901856, 1.5959758758544922, 1.2047193050384521]
# start_point = [0.553653400132947, -0.8120052615712687, 0.11354539249466497, -0.22812200929379317, -3.125237579486862, -0.05855107095312026]#tcp 16.5cm 8cm in y
# start_point2= [-0.797159496937887, -2.345598360101217, -1.536665439605713, -0.8599483531764527, 1.5955986976623535, 0.9196621775627136]
# start_point = [0.553676381105433, -0.8319837349953497, 0.11355234409748236, 0.0010960598943364995, -3.1363023705930035, -0.05928080589040827]#tcp 16.5cm 10cm in y
# start_point2= [-0.8112457434283655, -2.3661023579039515, -1.4952397346496582, -0.8811747592738648, 1.5952386856079102, 0.7590749859809875]
# start_point = [0.5936499896849933, -0.8319544378883524, 0.11353354742128832, 0.21567204793506817, -3.131423717440621, -0.05973227739189569]#tcp 12.5cm 10cm in y
# start_point2= [-0.782562557850973, -2.3948894939818324, -1.4373674392700195, -0.9096496862224122, 1.5961871147155762, 0.6508159637451172]
# start_point = [0.718010744900576, -0.8319892659550299, 0.1135682550727656, -2.420612047822881, -1.9510494320725742, -0.031052967702492707]#tcp 0cm 12cm in y
# start_point2= [-0.7022770086871546, -2.500256200829977, -1.224600076675415, -1.0150657457164307, 1.598847508430481, 2.653017520904541]
# start_point = [0.7179944815263763, -0.8520002327659106, 0.11360012784697238, -2.5283890649789185, -1.8074433598091864, -0.02806007137756851]#tcp 0cm 14cm in y
# start_point2= [-0.7160909811602991, -2.522921701470846, -1.1784803867340088, -1.0388308328441163, 1.5985084772109985, 2.754650354385376]
# start_point = [0.7179914754082709, -0.8719754891495058, 0.11356769992291205, -2.5283311228316894, -1.8074488214930953, -0.028050609716091583]#tcp 0cm 16cm in y
# start_point2= [-0.7295210997210901, -2.546648164788717, -1.1302876472473145, -1.0636816185763855, 1.598206639289856, 2.7410993576049805]
start_point = [0.7180010214182344, -0.8119068408012123, 0.11358855342237718, -2.430971729740251, -1.9380154621059815, -0.030744347697173845]#tcp 0cm 16cm in y
start_point2= [-0.6880562941180628, -2.47835697750234, -1.2690455913543701, -0.992119626407959, 1.5991392135620117, 2.678036689758301]


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
start_point2[5] = start_point2[5] - np.pi/2
# list_to_setp(setp, start_point, offset=12)
list_to_setp(setp, start_point2, offset=6)

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
max_angle = initial_joints[5] + np.pi

while attempts < max_attempts:
    try:
        joints_off = list(initial_joints)
        joints_off[5] += increment * (attempts + 1)

        print(f"joints_off[5] = {joints_off[5]:.4f}, max_angle = {max_angle:.4f}")

        if joints_off[5] > max_angle:
            break
        print(f"Attempt {attempts + 1}: Moving to joint position → {joints_off}")
        list_to_setp(setp, joints_off, offset=6)
        con.send(setp)
        time.sleep(0.2)

        while True:
            state = con.receive()
            if state is None:
                print("Lost connection while waiting.")
                break
            con.send(watchdog)
            if not state.output_bit_registers0_to_31:
                print("Movement completed.")
                break
        time.sleep(0.01)
        # Define output file
        if log == True:
            joint_state = state.actual_q
            tcp_state = state.actual_TCP_pose
            output_file = "results4.xlsx"
            new_capture()
            image_path = "/home/jack/literate-code/focused_image.jpg"
            pt1, pt2, angle = detect_red_points_and_angle(image_path)

            print("\n=== BEAM DETECTION ===")
            print("Red Point 1 (x,y):", pt1)
            print("Red Point 2 (x,y):", pt2)
            print("Beam angle (deg):", angle)
            # Format data as a dictionary
            data = {
                "Joint_1": [joint_state[0]],
                "Joint_2": [joint_state[1]],
                "Joint_3": [joint_state[2]],
                "Joint_4": [joint_state[3]],
                "Joint_5": [joint_state[4]],
                "Joint_6": [joint_state[5]],
                "x": [tcp_state[0]],
                "y": [tcp_state[1]],
                "z": [tcp_state[2]],
                "Beam_Angle_Deg": [angle]
            }

            new_row = pd.DataFrame(data)

            # Append to file or create new one
            if os.path.exists(output_file):
                existing_df = pd.read_excel(output_file)
                updated_df = pd.concat([existing_df, new_row], ignore_index=True)
            else:
                updated_df = new_row

            # Save to Excel
            updated_df.to_excel(output_file, index=False)

            print(f"\nSaved to Excel: {output_file}")
    except Exception as e:
        print(f"Error occurred: {e}")
        break

    attempts += 1

watchdog.input_int_register_0 = 3
con.send(watchdog) 
time.sleep(1.5) 

con.send_pause()
con.disconnect()
print("Disconnected")


