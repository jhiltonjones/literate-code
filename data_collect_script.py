import sys
import time
import logging
import math
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import rtde.rtde as rtde
import rtde.rtde_config as rtde_config

# --------- Camera hook (replace with your capture if different) ----------
# Expectation: new_capture() saves an image to IMAGE_PATH
from new_cam import new_capture
IMAGE_PATH = "/home/jack/literate-code/focused_image.jpg"  # adjust if needed

# --------- Beam angle helpers (your functions, lightly tidied) ----------
def compute_signed_angle(v1, v2):
    """Returns signed angle in degrees from v1 to v2 (CCW positive)."""
    angle1 = np.arctan2(v1[1], v1[0])
    angle2 = np.arctan2(v2[1], v2[0])
    angle_rad = angle2 - angle1
    angle_deg = np.degrees(angle_rad)
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
        (np.array([0, 50, 50]),   np.array([10, 255, 255])),
        (np.array([160, 50, 50]), np.array([180, 255, 255]))
    ]
    red_mask = None
    for lo, hi in red_ranges:
        m = cv2.inRange(image_hsv, lo, hi)
        red_mask = m if red_mask is None else cv2.bitwise_or(red_mask, m)

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
    reference = np.array([1, 0])  # x-axis to the right in image pixels
    angle = compute_signed_angle(reference, vector)

    if show:
        cv2.line(image, pt1, pt2, (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f"Beam Angle: {angle:.2f}°")
        plt.axis("off")
        plt.show()

    return pt1, pt2, angle

# --------- RTDE / Grid config ----------
ROBOT_HOST = "192.168.56.101"
ROBOT_PORT = 30004
CONFIG_XML = "control_loop_configuration.xml"
FREQUENCY = 25  # Hz streaming

JOINT_TARGET = [
-0.4509013334857386, -1.9217506847777308, -1.6537261009216309, -1.148660497074463, 1.538709282875061, -0.44603721300234014
]

GRID_STEP = 0.01    # meters between points in x,y
GRID_NX   = 3       # x columns
GRID_NY   = 3       # y rows
T_HOLD    = 1.5     # seconds dwell at each point

# Base TCP (z & orientation fixed)
TCP0 = [0.6863321291366229, -0.5298007815755799, 0.3988518734916411, 2.0782071104836115, -2.307145711170106, 0.0024772068811600738]

# Output file
RESULTS_XLSX = "grid_results.xlsx"

def write_joint_target(setp, q, offset=6):
    for i in range(6):
        setattr(setp, f"input_double_register_{offset + i}", float(q[i]))

def bit0_is_true(mask: int) -> bool:
    return (mask & 0b1) == 1

def main():
    logging.getLogger().setLevel(logging.INFO)

    # -- Setup RTDE
    conf = rtde_config.ConfigFile(CONFIG_XML)
    state_names, state_types = conf.get_recipe("state")
    setp_names, setp_types = conf.get_recipe("setp")
    watchdog_names, watchdog_types = conf.get_recipe("watchdog")

    con = rtde.RTDE(ROBOT_HOST, ROBOT_PORT)
    while con.connect() != 0:
        print("RTDE connect failed, retrying...")
        time.sleep(0.5)
    print("Connected.")

    con.get_controller_version()
    con.send_output_setup(state_names, state_types, FREQUENCY)
    setp = con.send_input_setup(setp_names, setp_types)
    watchdog = con.send_input_setup(watchdog_names, watchdog_types)

    # Init inputs
    for i in range(24):
        setattr(setp, f"input_double_register_{i}", 0.0)
    setp.input_bit_registers0_to_31 = 0
    watchdog.input_int_register_0 = 0

    if not con.send_start():
        print("Failed to send_start()")
        sys.exit(1)

    # Wait operator CONTINUE (Boolean output reg 0 → True)
    print("Waiting for operator CONTINUE on pendant (Boolean output reg 0 → True)...")
    while True:
        state = con.receive()
        if state is None:
            continue
        con.send(watchdog)
        if hasattr(state, "output_bit_registers0_to_31") and bit0_is_true(state.output_bit_registers0_to_31):
            break

    # MODE 1: moveJ safe start
    watchdog.input_int_register_0 = 1
    con.send(watchdog)
    write_joint_target(setp, JOINT_TARGET, offset=6)
    con.send(setp)
    time.sleep(0.2)

    # Wait moveJ finished (Boolean output reg 0 → False)
    while True:
        state = con.receive()
        if state is None:
            continue
        con.send(watchdog)
        if hasattr(state, "output_bit_registers0_to_31") and not bit0_is_true(state.output_bit_registers0_to_31):
            break

    # MODE 2: stream fixed-orientation XY grid (row-major)
    watchdog.input_int_register_0 = 2
    con.send(watchdog)
    DT = 1.0 / FREQUENCY

    # Center grid around (0,0)
    cx = (GRID_NX - 1) / 2.0
    cy = (GRID_NY - 1) / 2.0

    # Prepare output (create if missing)
    if not os.path.exists(RESULTS_XLSX):
        pd.DataFrame(columns=[
            "grid_ix","grid_iy","dx_m","dy_m",
            "joint_1","joint_2","joint_3","joint_4","joint_5","joint_6",
            "tcp_x","tcp_y","tcp_z","tcp_rx","tcp_ry","tcp_rz",
            "beam_pt1_x","beam_pt1_y","beam_pt2_x","beam_pt2_y","beam_angle_deg",
            "timestamp"
        ]).to_excel(RESULTS_XLSX, index=False)

    total = GRID_NX * GRID_NY
    idx = 0

    for iy in range(GRID_NY):          # row-major, no snake
        for ix in range(GRID_NX):
            idx += 1
            dx = (ix - cx) * GRID_STEP
            dy = (iy - cy) * GRID_STEP
            print(f"\nGrid {idx}/{total}  (ix,iy)=({ix},{iy})  offset: dx={dx:+.3f} m, dy={dy:+.3f} m")

            target = [
                TCP0[0] + dx,  # x
                TCP0[1] + dy,  # y
                TCP0[2],       # z fixed
                TCP0[3],       # rx fixed
                TCP0[4],       # ry fixed
                TCP0[5],       # rz fixed
            ]

            # Dwell at this target (stream same pose)
            t0 = time.perf_counter()
            next_tick = t0
            while (time.perf_counter() - t0) < T_HOLD:
                for i in range(6):
                    setattr(setp, f"input_double_register_{i}", float(target[i]))
                if "input_bit_registers0_to_31" in setp.__dict__:
                    setp.input_bit_registers0_to_31 = 0
                con.send(setp)

                state = con.receive()
                con.send(watchdog)

                next_tick += DT
                rem = next_tick - time.perf_counter()
                if rem > 0:
                    time.sleep(rem)

            # One more state read after dwell (steady)
            state = con.receive()
            con.send(watchdog)
            if state is None:
                print("Lost connection during logging; exiting.")
                break

            # Get actual joint angles & TCP pose
            joints = list(getattr(state, "actual_q"))
            tcp    = list(getattr(state, "actual_TCP_pose"))

            # Capture image & compute beam angle (robust to failures)
            beam_pt1 = beam_pt2 = (np.nan, np.nan)
            beam_angle = np.nan
            try:
                new_capture()  # writes IMAGE_PATH
                pt1, pt2, angle = detect_red_points_and_angle(IMAGE_PATH, show=False)
                beam_pt1, beam_pt2, beam_angle = pt1, pt2, float(angle)
                print(f"Beam angle: {beam_angle:.2f}°  | pt1={pt1} pt2={pt2}")
            except Exception as e:
                print(f"Beam detection failed at (ix={ix}, iy={iy}): {e}")

            # Append row to Excel
            row = pd.DataFrame([{
                "grid_ix": ix, "grid_iy": iy, "dx_m": dx, "dy_m": dy,
                "joint_1": joints[0], "joint_2": joints[1], "joint_3": joints[2],
                "joint_4": joints[3], "joint_5": joints[4], "joint_6": joints[5],
                "tcp_x": tcp[0], "tcp_y": tcp[1], "tcp_z": tcp[2],
                "tcp_rx": tcp[3], "tcp_ry": tcp[4], "tcp_rz": tcp[5],
                "beam_pt1_x": beam_pt1[0], "beam_pt1_y": beam_pt1[1],
                "beam_pt2_x": beam_pt2[0], "beam_pt2_y": beam_pt2[1],
                "beam_angle_deg": beam_angle,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }])

            # Load existing, append, save (Excel-friendly)
            try:
                existing = pd.read_excel(RESULTS_XLSX)
                out = pd.concat([existing, row], ignore_index=True)
            except Exception:
                out = row  # if file corrupted/locked, start fresh
            out.to_excel(RESULTS_XLSX, index=False)
            print(f"Logged grid point to {RESULTS_XLSX}")

            time.sleep(0.1)  # brief settle

    # MODE 3: halt/exit
    watchdog.input_int_register_0 = 3
    con.send(watchdog)
    print("\nXY grid + logging complete. Exiting.")

if __name__ == "__main__":
    main()
