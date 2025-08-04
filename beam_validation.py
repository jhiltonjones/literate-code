#!/usr/bin/env python3

import sys
import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config
from new_cam import new_capture  # Replace with your actual capture function if needed

# ─── ROBOT CONNECTION ─────────────────────────────────────────────────────
ROBOT_HOST = '192.168.56.101'
ROBOT_PORT = 30004
CONFIG_FILENAME = "control_loop_configuration.xml"

logging.getLogger().setLevel(logging.INFO)

conf = rtde_config.ConfigFile(CONFIG_FILENAME)
state_names, state_types = conf.get_recipe("state")
setp_names, setp_types = conf.get_recipe("setp")
watchdog_names, watchdog_types = conf.get_recipe("watchdog")

con = rtde.RTDE(ROBOT_HOST, ROBOT_PORT)
con.connect()

con.get_controller_version()

# Setup recipes
con.send_output_setup(state_names, state_types)
setp = con.send_input_setup(setp_names, setp_types)
watchdog = con.send_input_setup(watchdog_names, watchdog_types)

setp.input_double_register_0 = 0
setp.input_double_register_1 = 0
setp.input_double_register_2 = 0
setp.input_double_register_3 = 0
setp.input_double_register_4 = 0
setp.input_double_register_5 = 0
watchdog.input_int_register_0 = 0

if not con.send_start():
    sys.exit()

# ─── GET ROBOT STATE ONCE ────────────────────────────────────────────────
state = con.receive()
if state is None:
    print("Failed to receive robot state")
    sys.exit()

joint_state = state.actual_q
tcp_pose = state.actual_TCP_pose

print("\n=== ROBOT STATE ===")
print("Joint Positions:", joint_state)
print("TCP Pose (x, y, z, rx, ry, rz):", tcp_pose)

# ─── IMAGE PROCESSING ────────────────────────────────────────────────────
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

def detect_red_points_and_angle(image_path, show=True):
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

if __name__ == "__main__":
    # ─── IMAGE INPUT AND ANGLE COMPUTATION ───────────────────────────────────
    new_capture()
    image_path = "/home/jack/literate-code/focused_image.jpg"
    pt1, pt2, angle = detect_red_points_and_angle(image_path)

    print("\n=== BEAM DETECTION ===")
    print("Red Point 1 (x,y):", pt1)
    print("Red Point 2 (x,y):", pt2)
    print("Beam angle (deg):", angle)

    # ─── CLEANUP ──────────────────────────────────────────────────────────────
    con.send_pause()
    con.disconnect()


    # Define output file
    output_file = "results.xlsx"

    # Format data as a dictionary
    data = {
        "Joint_1": [joint_state[0]],
        "Joint_2": [joint_state[1]],
        "Joint_3": [joint_state[2]],
        "Joint_4": [joint_state[3]],
        "Joint_5": [joint_state[4]],
        "Joint_6": [joint_state[5]],
        "x": [tcp_pose[0]],
        "y": [tcp_pose[1]],
        "z": [tcp_pose[2]],
        "Beam_Angle_Deg": [angle]
    }

    new_row = pd.DataFrame(data)

    # Append to file or create new one
    # if os.path.exists(output_file):
    #     existing_df = pd.read_excel(output_file)
    #     updated_df = pd.concat([existing_df, new_row], ignore_index=True)
    # else:
    #     updated_df = new_row

    # # Save to Excel
    # updated_df.to_excel(output_file, index=False)

    # print(f"\nSaved to Excel: {output_file}")
start_point_16_5_2_tcp = [-0.7359426657306116, -2.2689033947386683, -1.6913682222366333, -0.780555323963501, 1.5971418619155884, 1.999606728553772]#tcp 16.5cm
start_point_16_5_2_j = [0.5536488450634438, -0.7320652960373009, 0.1135122420106864, -1.71332968251415, -2.6031873873441627, -0.045181316562570746]
start_point_16_5_4_tcp = [0.5536390628501511, -0.7520670276634125, 0.11350151688715632, -1.7133290981850664, -2.6032191894953556, -0.04515480646310208]#tcp 16.5cm
start_point_16_5_4_j= [-0.7520225683795374, -2.28737272838735, -1.6542009115219116, -0.7996226114085694, 1.5967273712158203, 1.983452320098877]
