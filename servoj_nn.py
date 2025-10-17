# beam
# Joint State: [-0.3329866568194788, -2.0541001758971156, -1.0448529720306396, -1.6290365658202113, 1.5409388542175293, 0.4389771819114685]
# TCP Pose:  [0.8626541834323219, -0.4868659841674622, 0.6304371536197035, 1.206710674372489, -2.8637625212082796, 0.021032903808600848]
# robot
# Joint State: [-0.45086700121034795, -1.9217144451537074, -1.6537175178527832, -1.1486450296691437, 1.5386974811553955, 0.32281869649887085]
# TCP Pose:  [0.6863470071188493, -0.5297771147199332, 0.3988899776980005, 1.2068194033944717, -2.8637413293536538, 0.02102025865000871]
import sys
import time
import logging
from new_cam import new_capture
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import csv, shutil
import torch
from neural_net import SimpleMLP
from nn_test import solve_joint6_for_angle

class AngleUnwrapper:
    def __init__(self): self.prev = None
    def __call__(self, a):
        if self.prev is None:
            self.prev = a
            return a
        delta = a - self.prev
        if   delta > 180.0: a -= 360.0
        elif delta < -180.0: a += 360.0
        self.prev = a
        return a

unwrap_angle = AngleUnwrapper()

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

    # after you get red_centers
    red_centers.sort(key=lambda p: (p[0], p[1]))  # sort by x then y
    pt1, pt2 = red_centers  # pt1 = leftmost, pt2 = rightmost

    vector = np.array(pt2) - np.array(pt1)
    reference = np.array([1.0, 0.0])
    raw_angle = compute_signed_angle(reference, vector)  # in degrees
    angle = unwrap_angle(raw_angle)                      # use this instead of raw_angle


    if show:
        cv2.line(image, pt1, pt2, (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f"Beam Angle: {angle:.2f}°")
        plt.axis("off")
        plt.show()

    return pt1, pt2, angle
RUN_ROOT = Path("NN_control")
RUN_DIR = RUN_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR.mkdir(parents=True, exist_ok=True)
print(f"[LOG] Saving outputs to: {RUN_DIR}")

def list_to_setp(setp, lst, offset=0):
    """ Converts a list into RTDE input registers, allowing different sets with offsets. """
    for i in range(6):
        setp.__dict__[f"input_double_register_{i + offset}"] = lst[i]
    return setp

ROBOT_HOST = "192.168.56.101"
ROBOT_PORT = 30004
CONFIG_XML = "control_loop_configuration.xml"
FREQUENCY = 25  # use 125 if your controller prefers it

# input_double_registers 6..11 hold the joint target q[0..5]
JOINT_TARGET = [
-0.45088225999941045, -1.9217144451537074, -1.6537089347839355, -1.148652271633484, 1.538681983947754, 0.9267773628234863
]
JOINT_TARGET2 = [
-0.4109237829791468, -1.8232914410033167, -1.5675734281539917, -1.3344539117864151, 1.5394864082336426, -3.662370030079977
]
JOINT_TARGET3 = [
    -0.4110644499408167,
    -1.8883592091002406,
    -1.699528455734253,
    -1.1374167960933228,
    1.5391621589660645,
    -3.4619577566729944,
]
def main():
    logging.getLogger().setLevel(logging.INFO)

    # Load RTDE recipes
    conf = rtde_config.ConfigFile(CONFIG_XML)
    state_names, state_types = conf.get_recipe("state")
    setp_names, setp_types = conf.get_recipe("setp")
    watchdog_names, watchdog_types = conf.get_recipe("watchdog")

    # Connect
    con = rtde.RTDE(ROBOT_HOST, ROBOT_PORT)
    while con.connect() != 0:
        print("RTDE connect failed, retrying...")
        time.sleep(0.5)
    print("Connected.")

 
    con.get_controller_version()

    # Setup streams
    con.send_output_setup(state_names, state_types, FREQUENCY)
    setp = con.send_input_setup(setp_names, setp_types)
    watchdog = con.send_input_setup(watchdog_names, watchdog_types)

    # Clear inputs
    for i in range(24):
        setattr(setp, f"input_double_register_{i}", 0.0)
    setp.input_bit_registers0_to_31 = 0
    watchdog.input_int_register_0 = 0

    if not con.send_start():
        print("Failed to send_start()")
        sys.exit(1)

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

    list_to_setp(setp, JOINT_TARGET, offset=6)

    con.send(setp)
    time.sleep(0.5)

    while True:
        print(f'Waiting for moveJ()1 to finish start...')
        state = con.receive()
        con.send(watchdog)
        if not state.output_bit_registers0_to_31:
            print('Start completed, proceeding to feedback check\n')
            break

    # --- MODE 2: servoj streaming with NN controller (regs 6..11) ---
    watchdog.input_int_register_0 = 2
    con.send(watchdog)
    _ = con.receive()  # prime receive loop

    # Load NN once
    model = SimpleMLP()
    model.load_state_dict(torch.load("simple_mlp.pt", map_location="cpu"))
    model.eval()

    # Logs
    t_log, ref_deg_log, meas_deg_log, err_deg_log, j6_rad_log = [], [], [], [], []

    # Reference (deg)
    A_deg, bias_deg, freq_hz, phase_deg, duration_s = 15.0, 0.0, 0.02, 0.0, 60.0
    A_rad     = np.deg2rad(A_deg)
    bias_rad  = np.deg2rad(bias_deg)
    phase_rad = np.deg2rad(phase_deg)

    # Stream cadence must match servoj t
    dt = 1.0 / FREQUENCY

    joint_pos = JOINT_TARGET.copy()

    # Safety: soft rate limit & joint bounds
    MAX_STEP = np.deg2rad(12.0)     # allow up to 2° per cycle; raise slowly if stable
    J6_MIN   = -np.inf
    J6_MAX   = +np.inf

    def clamp_step(curr, target, max_step):
        d = target - curr
        if d > max_step:  return curr + max_step
        if d < -max_step: return curr - max_step
        return target

    t0 = time.time()
    next_tick = time.perf_counter()

    while True:
        t = time.time() - t0
        if t > duration_s:
            break

        # 1) Measure current angle (deg)
        new_capture()
        image_path = "/home/jack/literate-code/focused_image.jpg"
        _, _, angle_deg = detect_red_points_and_angle(image_path)

        # 2) Reference (deg + rad)
        ref_rad = bias_rad + A_rad * np.sin(2.0 * np.pi * freq_hz * t + phase_rad)
        ref_deg = float(np.rad2deg(ref_rad))
        err_deg = ref_deg - angle_deg

        # 3) NN inverse: desired angle -> joint6 (rad)
        #    (Assumes your solve_joint6_for_angle uses 'model' to invert)
        j6_sol, ok = solve_joint6_for_angle(
            model, target_deg=ref_deg, j_min=J6_MIN, j_max=J6_MAX, x0=joint_pos[5]
        )
        # Clamp to limits and rate-limit per cycle
        j6_cmd = float(np.clip(j6_sol, J6_MIN, J6_MAX))
        j6_cmd = clamp_step(joint_pos[5], j6_cmd, MAX_STEP)

        # 4) Send new joint target via RTDE (regs 6..11)
        joint_pos[5] = j6_cmd
        list_to_setp(setp, joint_pos, offset=6)
        if "input_bit_registers0_to_31" in setp_names:
            setp.input_bit_registers0_to_31 = 0
        con.send(setp)

        # 5) Keepalive + precise pacing (servoj(t=0.04) <-> dt=0.04)
        _ = con.receive()
        con.send(watchdog)
        next_tick += dt
        rem = next_tick - time.perf_counter()
        if rem > 0:
            time.sleep(rem)

        # 6) Log
        print(f"t={t:5.2f}s ref={ref_deg:6.2f}° meas={angle_deg:6.2f}° "
            f"err={err_deg:6.2f}° j6={joint_pos[5]:.3f} rad [{'ok' if ok else 'extr'}]")
        t_log.append(t); ref_deg_log.append(ref_deg); meas_deg_log.append(angle_deg)
        err_deg_log.append(err_deg); j6_rad_log.append(joint_pos[5])

    # --- end NN streaming ---

    # exit Mode 2
    watchdog.input_int_register_0 = 3
    con.send(watchdog)

    print("Mode 3 sent — robot should move to Halt section now.")
    time.sleep(0.1)
    con.send_pause()
    con.disconnect()

    # Save figures
    ( RUN_DIR / "plots" ).mkdir(exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(t_log, ref_deg_log, label="Reference (deg)", linewidth=2)
    plt.plot(t_log, meas_deg_log, label="Measured (deg)", linewidth=1.5)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlabel("Time (s)"); plt.ylabel("Angle (deg)")
    plt.title("Sine Tracking: Reference vs Measured"); plt.legend(); plt.tight_layout()
    plt.savefig(RUN_DIR / "plots" / "angles_ref_meas.png", dpi=200)

    plt.figure(figsize=(10, 4))
    plt.plot(t_log, err_deg_log, label="Error (deg)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlabel("Time (s)"); plt.ylabel("Error (deg)")
    plt.title("Tracking Error"); plt.legend(); plt.tight_layout()
    plt.savefig(RUN_DIR / "plots" / "error.png", dpi=200)


    plt.figure(figsize=(10, 4))
    plt.plot(t_log, j6_rad_log, label="Joint 6 (rad)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlabel("Time (s)"); plt.ylabel("Joint 6 (rad)")
    plt.title("Joint 6 Command"); plt.legend(); plt.tight_layout()
    plt.savefig(RUN_DIR / "plots" / "j6.png", dpi=200)

    plt.show()
    print(f"[LOG] Plots saved under {RUN_DIR/'plots'}")


if __name__ == "__main__":
    main()
