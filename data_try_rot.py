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

# --------- Camera hook ----------
from new_cam import new_capture
IMAGE_PATH = "/home/jack/literate-code/focused_image.jpg"  # adjust if needed


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
JOINT_TARGET = [
-0.39337331453432256, -1.989443918267721, -1.5559700727462769, -1.1809083384326478, 1.539546012878418, 1.6148388385772705]
GRID_STEP = 0.0    # meters
GRID_NX   = 1
GRID_NY   = 1
T_HOLD    = 0.5    # seconds per pose (reduce to fit more data)

# Base TCP (z & base orientation)

# TCP0 = [0.7444982728931748, -0.5284124861014445, 0.39884576718636294, 0.3881354040517943, -3.0881379446948745, 0.034401442335245426] perpendicular
# TCP0 = [0.7477774659086398, -0.6007309595177756, 0.39883841647415963, 1.305924149254298, -2.8194402999111485, 0.01905514958953238]
# TCP0 = [0.7613563761169214, -0.6323676512264372, 0.3988764263237069, 1.657680313149333, -2.6266170486132676, 0.012076413305251487]
TCP0 = [0.7573552061757611, -0.506778797150568, 0.3987371972414318, -0.6765979864161791, -3.0469228618765967, 0.04772737453020368]

# --- Simple 2D rotations to test (yaw about world Z, degrees) ---
YAW_MIN_DEG = -90
YAW_MAX_DEG =  90
YAW_STEP_DEG = 2
YAW_DEG_LIST = list(range(YAW_MIN_DEG, YAW_MAX_DEG + 1, YAW_STEP_DEG))
Y_CENTER_SHIFT = 0.0 # meters; +0.02 shifts the grid 2 cm in +X


# Output file
RESULTS_XLSX = "grid_results_friday.xlsx"

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

    # ===== Mode 2: stream XY + yaw (2D) =====
    watchdog.input_int_register_0 = 2
    con.send(watchdog)
    DT = 1.0 / FREQUENCY

    # Centered grid
    cx = (GRID_NX - 1) / 2.0
    cy = (GRID_NY - 1) / 2.0

    # Prepare output once
    if not os.path.exists(RESULTS_XLSX):
        pd.DataFrame(columns=[
            "grid_ix","grid_iy","dx_m","dy_m","yaw_deg",
            "joint_1","joint_2","joint_3","joint_4","joint_5","joint_6",
            "tcp_x","tcp_y","tcp_z","tcp_rx","tcp_ry","tcp_rz",
            "beam_pt1_x","beam_pt1_y","beam_pt2_x","beam_pt2_y","beam_angle_deg",
            "timestamp"
        ]).to_excel(RESULTS_XLSX, index=False)

    # Base orientation matrix from TCP0 axis-angle
    R0 = axis_angle_to_rot(TCP0[3], TCP0[4], TCP0[5])

    # Simple yaw timing rule
    YAW_JUMP_THRESHOLD_DEG = 10.0
    BIG_JUMP_TRANSIT_SEC   = 10.0  # slow transition duration for big jumps

    total_cells = GRID_NX * GRID_NY * len(YAW_DEG_LIST)
    done = 0
    prev_yaw = 12  # deg

    for iy in range(GRID_NY):
        for ix in range(GRID_NX):
            dx = (ix - cx) * GRID_STEP
            dy = (iy - cy) * GRID_STEP

            for yaw_deg in YAW_DEG_LIST:
                done += 1

                # Tool-frame yaw to reduce IK flips: R = R0 @ Rz(yaw)
                yaw_rad = math.radians(yaw_deg)
                R = R0 @ Rz(yaw_rad)
                rx, ry, rz = rot_to_axis_angle(R)

                target = [TCP0[0]  + dx, TCP0[1] + Y_CENTER_SHIFT + dy, TCP0[2], rx, ry, rz]

                print(f"\n[{done}/{total_cells}] (ix,iy)=({ix},{iy}) dx={dx:+.3f} dy={dy:+.3f} | yaw={yaw_deg:+.1f}°")

                # ------------ slow only big yaw jumps (>10°) over 10 seconds ------------
                if prev_yaw is not None:
                    dyaw = yaw_deg - prev_yaw
                    if abs(dyaw) > YAW_JUMP_THRESHOLD_DEG:
                        steps = max(1, int(BIG_JUMP_TRANSIT_SEC / DT))
                        for k in range(steps + 1):
                            y_interp = prev_yaw + dyaw * (k / steps)
                            R_mid = R0 @ Rz(math.radians(y_interp))
                            rmx, rmy, rmz = rot_to_axis_angle(R_mid)
                            mid_pose = [TCP0[0] + dx, TCP0[1] + dy, TCP0[2], rmx, rmy, rmz]
                            # stream one tick at this intermediate yaw
                            for j in range(6):
                                setattr(setp, f"input_double_register_{j}", float(mid_pose[j]))
                            if "input_bit_registers0_to_31" in setp.__dict__:
                                setp.input_bit_registers0_to_31 = 0
                            con.send(setp)
                            _ = con.receive(); con.send(watchdog)
                            time.sleep(DT)
                # ------------------------------------------------------------------------

                # Dwell at final target pose for T_HOLD
                t0 = time.perf_counter(); next_tick = t0
                while (time.perf_counter() - t0) < T_HOLD:
                    for j in range(6):
                        setattr(setp, f"input_double_register_{j}", float(target[j]))
                    if "input_bit_registers0_to_31" in setp.__dict__:
                        setp.input_bit_registers0_to_31 = 0
                    con.send(setp)
                    state = con.receive(); con.send(watchdog)
                    next_tick += DT
                    rem = next_tick - time.perf_counter()
                    if rem > 0: time.sleep(rem)

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
                    print(f"Beam detection failed (ix={ix}, iy={iy}, yaw={yaw_deg}): {e}")

                # Append row
                row = pd.DataFrame([{
                    "grid_ix": ix, "grid_iy": iy, "dx_m": dx, "dy_m": dy, "yaw_deg": yaw_deg,
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

                # remember last yaw
                prev_yaw = yaw_deg

    # ===== Mode 3: halt/exit =====
    watchdog.input_int_register_0 = 3
    con.send(watchdog)
    print("\nXY grid with 2D yaw rotations complete. Exiting.")


if __name__ == "__main__":
    main()
