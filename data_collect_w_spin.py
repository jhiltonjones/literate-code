#Joint State: [-0.3063767592059534, -2.1076585255064906, -0.702275276184082, -1.9189321003355921, 1.5413981676101685, -0.5084031263934534]
#TCP Pose:  [0.8730334395676191, -0.4632732173554521, 0.7761266586348291, 2.4037096158647064, -1.9660304862843407, -0.006136965176345673]
import sys
import time
import logging
import math
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from new_cam import detect_red_points_and_angle
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config
from transformations import get_point
# --------- Camera hook ----------
from new_cam import new_capture
IMAGE_PATH = "/home/jack/literate-code/focused_image.jpg"  # adjust if needed
import numpy as np
import math

def rotz(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])

def roty(theta):
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0,             1, 0],
        [-np.sin(theta),0, np.cos(theta)]
    ])

def rotx(theta):
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)]
    ])

def transGen(R, t):
    H = np.eye(4)
    H[:3, :3] = R
    H[:3,  3] = t
    return H

def rotvec_to_R(r):
    r = np.asarray(r)
    theta = np.linalg.norm(r)
    if theta < 1e-9:
        return np.eye(3)
    k = r / theta
    kx, ky, kz = k
    K = np.array([
        [0,    -kz,   ky],
        [kz,    0,   -kx],
        [-ky,  kx,    0]
    ])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R

def R_to_rotvec(R):
    R = np.asarray(R)
    tr = np.trace(R)
    cos_theta = (tr - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-9:
        return np.zeros(3)

    rx = (R[2,1] - R[1,2]) / (2*np.sin(theta))
    ry = (R[0,2] - R[2,0]) / (2*np.sin(theta))
    rz = (R[1,0] - R[0,1]) / (2*np.sin(theta))
    k = np.array([rx, ry, rz])
    return theta * k

def rotate_around_point_transform(axis, pivot_pos, theta):
    if axis in ('x', 1):
        R = rotx(theta)
    elif axis in ('y', 2):
        R = roty(theta)
    else:
        R = rotz(theta)

    c = np.asarray(pivot_pos).reshape(3) 
    I = np.eye(3)
    t = (I - R) @ c
    return transGen(R, t)

def get_point(theta_angle_x, theta_angle_z):
    start_point = np.array([
        0.8044738038441734, -0.5510007927198923, 0.4330498000582408,
        -2.29598721437706,  2.137233721135031,   0.02151272219562204
    ])

    pivot_point = np.array([
        0.8044738038441734, -0.4112657614330756, 0.17238122113144055,
        -3.0760125513655026, -0.5854704390599933, 0.08213552142709286
    ])

    ee_pos0   = start_point[:3]
    ee_rvec0  = start_point[3:]
    R_b_e0    = rotvec_to_R(ee_rvec0)
    H_b_e0    = transGen(R_b_e0, ee_pos0)

    mag_offset = np.array([0, 0, 0.2])
    H_e_m      = transGen(np.eye(3), mag_offset)

    H_b_m0     = H_b_e0 @ H_e_m
    pivot_pos  = pivot_point[:3]

    theta_z = np.deg2rad(theta_angle_z)
    H_rot_z = rotate_around_point_transform('z', pivot_pos, theta_z)

    theta_x = np.deg2rad(theta_angle_x)
    H_rot_x = rotate_around_point_transform('x', pivot_pos, theta_x)

    H_rot   = H_rot_x @ H_rot_z
    H_b_m1  = H_rot @ H_b_m0

    H_m_e   = np.linalg.inv(H_e_m)
    H_b_e1  = H_b_m1 @ H_m_e

    new_pos  = H_b_e1[:3, 3]
    new_R    = H_b_e1[:3, :3]
    new_rvec = R_to_rotvec(new_R)

    new_pose_for_robot = np.hstack([new_pos, new_rvec])
    return new_pose_for_robot


def get_point_with_spin(theta_angle_x, theta_angle_z, theta_spin_z):
    """
    theta_angle_x  : outer rotation about x (deg)
    theta_angle_z  : outer rotation about z (deg) using your main pivot
    theta_spin_z   : inner rotation about z (deg) around the magnet's own center
    """
    # ---- first: same as your get_point() ----
    start_point = np.array([
        0.8044738038441734, -0.5510007927198923, 0.4330498000582408,
        -2.29598721437706,  2.137233721135031,   0.02151272219562204
    ])

    pivot_point = np.array([
        0.8044738038441734, -0.4112657614330756, 0.17238122113144055,
        -3.0760125513655026, -0.5854704390599933, 0.08213552142709286
    ])

    ee_pos0   = start_point[:3]
    ee_rvec0  = start_point[3:]
    R_b_e0    = rotvec_to_R(ee_rvec0)
    H_b_e0    = transGen(R_b_e0, ee_pos0)

    mag_offset = np.array([0, 0, 0.2])
    H_e_m      = transGen(np.eye(3), mag_offset)

    H_b_m0     = H_b_e0 @ H_e_m
    pivot_pos  = pivot_point[:3]

    theta_z = np.deg2rad(theta_angle_z)
    H_rot_z = rotate_around_point_transform('z', pivot_pos, theta_z)

    theta_x = np.deg2rad(theta_angle_x)
    H_rot_x = rotate_around_point_transform('x', pivot_pos, theta_x)

    H_rot   = H_rot_x @ H_rot_z
    H_b_m1  = H_rot @ H_b_m0

    # ---- second: spin around magnet's own center, about z ----
    # current magnet center in base frame
    magnet_center = H_b_m1[:3, 3]
    theta_spin = np.deg2rad(theta_spin_z)
    H_spin_z   = rotate_around_point_transform('z', magnet_center, theta_spin)

    H_b_m2 = H_spin_z @ H_b_m1

    # back to EE frame
    H_m_e   = np.linalg.inv(H_e_m)
    H_b_e2  = H_b_m2 @ H_m_e

    new_pos  = H_b_e2[:3, 3]
    new_R    = H_b_e2[:3, :3]
    new_rvec = R_to_rotvec(new_R)

    new_pose_for_robot = np.hstack([new_pos, new_rvec])
    return new_pose_for_robot


# --------- Axis-angle <-> rotation helpers (for composing yaw) ----------
def axis_angle_to_rot(rx, ry, rz):
    theta = math.sqrt(rx*rx + ry*ry + rz*rz)
    if theta < 1e-12:
        return np.eye(3)
    kx, ky, kz = rx/theta, ry/theta, rz/theta
    K = np.array([[0, -kz, ky],[kz, 0, -kx],[-ky, kx, 0]], dtype=float)
    R = np.eye(3) + math.sin(theta)*K + (1.0 - math.cos(theta))*(K @ K)
    return R

def rot_to_axis_angle(R):
    eps = 1e-12
    tr = np.trace(R)
    # Clamp for numerical stability
    c = (tr - 1.0) / 2.0
    c = max(min(c, 1.0), -1.0)
    theta = math.acos(c)
    if theta < 1e-12:
        return (0.0, 0.0, 0.0)
    rx = (R[2,1] - R[1,2]) / (2*math.sin(theta))
    ry = (R[0,2] - R[2,0]) / (2*math.sin(theta))
    rz = (R[1,0] - R[0,1]) / (2*math.sin(theta))
    return (theta*rx, theta*ry, theta*rz)

def Rz(yaw_rad):
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.array([[c,-s,0],[s, c,0],[0,0,1]], dtype=float)

# --------- RTDE / Grid config ----------
ROBOT_HOST = "192.168.56.101"
ROBOT_PORT = 30004
CONFIG_XML = "control_loop_configuration.xml"
FREQUENCY = 25  # Hz streaming

# JOINT_TARGET = [
# -0.4901054541217249, -2.0630060635008753, -1.4456350803375244, -1.2141221326640625, 1.5384504795074463, 0.2132866382598877
# ]
# JOINT_TARGET = [
# -0.5121501127826136, -2.1110855541624964, -1.3705785274505615, -1.2402780813029786, 1.5382890701293945, -0.06735116640199834
# ]
JOINT_TARGET = [-0.4221299330340784, -2.071312566796774, -1.3693373203277588, -1.282651738529541, 1.5822639465332031, -0.4940570036517542]

GRID_STEP = 0.01    # meters
GRID_NX   = 5
GRID_NY   = 1
T_HOLD    = 5    # seconds per pose (reduce to fit more data)

# Base TCP (z & base orientation)


TCP0 = [0.7844738038441734, -0.5510007927198923, 0.4330498000582408, -2.29598721437706, 2.137233721135031, 0.02151272219562204]

# --- Simple 2D rotations to test (yaw about world Z, degrees) ---
YAW_MIN_DEG = -20
YAW_MAX_DEG =  20
YAW_STEP_DEG = 5
YAW_DEG_LIST = list(range(YAW_MIN_DEG, YAW_MAX_DEG + 1, YAW_STEP_DEG))
Y_CENTER_SHIFT = 0.00 # meters; +0.02 shifts the grid 2 cm in +X


# Output file
RESULTS_XLSX = "grid_results_testing.xlsx"

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

    # Wait operator CONTINUE
    print("Waiting for operator CONTINUE on pendant (Boolean output reg 0 → True)...")
    while True:
        state = con.receive()
        if state is None:
            continue
        con.send(watchdog)
        if hasattr(state, "output_bit_registers0_to_31") and bit0_is_true(state.output_bit_registers0_to_31):
            break

    # Mode 1: moveJ to start
    watchdog.input_int_register_0 = 1
    con.send(watchdog)
    write_joint_target(setp, JOINT_TARGET, offset=6)
    write_joint_target(setp, TCP0, offset=0)
    con.send(setp)
    time.sleep(0.2)

    # Wait for moveJ complete
    while True:
        state = con.receive()
        if state is None:
            continue
        con.send(watchdog)
        if hasattr(state, "output_bit_registers0_to_31") and not bit0_is_true(state.output_bit_registers0_to_31):
            break

    # ===== Mode 2: outer yaw (-90..90), inner spin (-80..80) =====
    watchdog.input_int_register_0 = 2
    con.send(watchdog)
    DT = 1.0 / FREQUENCY

    YAW_JUMP_THRESHOLD_DEG = 1
    BIG_JUMP_TRANSIT_SEC   = 10.0  # for smoothing big yaw changes

    SPIN_MIN_DEG = -20
    SPIN_MAX_DEG =  20
    SPIN_STEP_DEG = 3
    SPIN_DEG_LIST = list(range(SPIN_MIN_DEG, SPIN_MAX_DEG + 1, SPIN_STEP_DEG))

    total_steps = len(YAW_DEG_LIST) * len(SPIN_DEG_LIST)
    done = 0
    prev_yaw = 12

    for yaw_deg in YAW_DEG_LIST:
        # -------- smooth outer yaw transitions using get_point_with_spin(.., spin=0) ----
        if prev_yaw is not None:
            dyaw = yaw_deg - prev_yaw
            if abs(dyaw) > YAW_JUMP_THRESHOLD_DEG:
                steps = max(1, int(BIG_JUMP_TRANSIT_SEC / DT))
                for k in range(steps + 1):
                    y_interp = prev_yaw + dyaw * (k / steps)
                    mid_pose = get_point_with_spin(0, y_interp, 0)  # spin 0 during transit

                    for j in range(6):
                        setattr(setp, f"input_double_register_{j}", float(mid_pose[j]))
                    if "input_bit_registers0_to_31" in setp.__dict__:
                        setp.input_bit_registers0_to_31 = 0
                    con.send(setp)
                    _ = con.receive(); con.send(watchdog)
                    time.sleep(DT)
        # -------------------------------------------------------------------------------

        for spin_deg in SPIN_DEG_LIST:
            done += 1
            print(f"\n[{done}/{total_steps}] yaw(z)={yaw_deg:+.1f}°, spin(z)={spin_deg:+.1f}°")

            # Pose from full kinematic model: outer yaw + local spin
            target = get_point_with_spin(0, yaw_deg, spin_deg)

            # Dwell at final pose for T_HOLD
            t0 = time.perf_counter()
            next_tick = t0
            while (time.perf_counter() - t0) < T_HOLD:
                for j in range(6):
                    setattr(setp, f"input_double_register_{j}", float(target[j]))
                if "input_bit_registers0_to_31" in setp.__dict__:
                    setp.input_bit_registers0_to_31 = 0
                con.send(setp)
                state = con.receive(); con.send(watchdog)
                next_tick += DT
                rem = next_tick - time.perf_counter()
                if rem > 0:
                    time.sleep(rem)

            # Final state read for logging
            state = con.receive(); con.send(watchdog)
            if state is None:
                print("Lost connection during logging; exiting.")
                break

            joints = list(getattr(state, "actual_q"))
            tcp    = list(getattr(state, "actual_TCP_pose"))

            # Beam capture
            beam_pt1 = beam_pt2 = (np.nan, np.nan)
            beam_angle = np.nan
            try:
                new_capture()
                pt1, pt2, angle = detect_red_points_and_angle(IMAGE_PATH, show=False)
                beam_pt1, beam_pt2, beam_angle = pt1, pt2, float(angle)
                print(f"Beam angle: {beam_angle:.2f}°  | pt1={pt1} pt2={pt2}")
            except Exception as e:
                print(f"Beam detection failed (yaw={yaw_deg}, spin={spin_deg}): {e}")

            # Append row to Excel
            row = pd.DataFrame([{
                "grid_ix": 0, "grid_iy": 0, "dx_m": 0.0, "dy_m": 0.0,
                "yaw_deg": yaw_deg,
                "spin_deg": spin_deg,
                "joint_1": joints[0], "joint_2": joints[1], "joint_3": joints[2],
                "joint_4": joints[3], "joint_5": joints[4], "joint_6": joints[5],
                "tcp_x": tcp[0], "tcp_y": tcp[1], "tcp_z": tcp[2],
                "tcp_rx": tcp[3], "tcp_ry": tcp[4], "tcp_rz": tcp[5],
                "beam_pt1_x": beam_pt1[0], "beam_pt1_y": beam_pt1[1],
                "beam_pt2_x": beam_pt2[0], "beam_pt2_y": beam_pt2[1],
                "beam_angle_deg": beam_angle,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }])

            try:
                existing = pd.read_excel(RESULTS_XLSX)
                out = pd.concat([existing, row], ignore_index=True)
            except Exception:
                out = row
            out.to_excel(RESULTS_XLSX, index=False)

        prev_yaw = yaw_deg



    # ===== Mode 3: halt/exit =====
    watchdog.input_int_register_0 = 3
    con.send(watchdog)
    print("\nXY grid with 2D yaw rotations complete. Exiting.")


if __name__ == "__main__":
    main()
