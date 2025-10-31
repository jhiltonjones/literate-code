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
from vessel_trajecotry_plot import make_ref_tortuous

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
        (np.array([0,   120, 120]), np.array([8,   255, 240])),  # lower red band
        (np.array([172, 120, 120]), np.array([180, 255, 240]))   # upper red band
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
-0.41963416734804326, -1.9172355137267054, -1.659855604171753, -1.1482085150531312, 1.539107322692871, 0.932235360145569
]
# JOINT_TARGET = [
# -0.41963416734804326, -1.9172355137267054, -1.659855604171753, -1.1482085150531312, 1.539107322692871, 0.8993573188781738]

# JOINT_TARGET = [
# -0.41963416734804326, -1.9172355137267054, -1.659855604171753, -1.1482085150531312, 1.539107322692871, -0.254600826893942
# ]



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

    watchdog.input_int_register_0 = 2
    con.send(watchdog)
    state = con.receive()
    # logs
    # --- before the loop, load the forward NN once ---
    model = SimpleMLP()
    model.load_state_dict(torch.load("simple_mlp.pt", map_location="cpu"))
    model.eval()

    # logs
    t_log, ref_deg_log, meas_deg_log = [], [], []
    err_deg_log, j6_rad_log = [], []

    # sine ref params
    # A_deg, bias_deg, freq_hz, phase_deg, duration_s = 15.0, 0.0, 0.02, 0.0, 60.0
    A_deg, bias_deg, freq_hz, phase_deg, duration_s = 15.0, 45.0, 2.5, 0.0, .5

    A_rad     = np.deg2rad(A_deg)
    bias_rad  = np.deg2rad(bias_deg)
    phase_rad = np.deg2rad(phase_deg)
    # dt = 0.1
    dt=0.005
    # joint 6 limits (rad)
    # J6_MIN = JOINT_TARGET[5] - np.pi/2
    # J6_MAX = JOINT_TARGET[5] + np.pi/2

    joint_pos = JOINT_TARGET.copy()

    t0 = time.time()
    # while True:
    #     t = time.time() - t0
    #     if t > duration_s:
    #         break
    ref = make_ref_tortuous(A_rad=0.5, f_slow=2.4, f_fast=6.5, offset_rad=0.0, phase_rad=0.0)

    N = int(round(duration_s/dt))
    for k in range(N):
        t = k*dt
        # 1) measure current angle


        # 2) reference (both rad & deg, solver needs deg)
        # ref_rad = bias_rad + A_rad * np.sin(2.0 * np.pi * freq_hz * t + phase_rad)
        # ref_deg = float(np.rad2deg(ref_rad))
        theta_ref, theta_refdot = ref(t)   # tuple: (angle, angular rate)

        # === this replaces your old line ===
        ref_rad = theta_ref
        # 3) error (for logging only)
        ref_deg = float(np.rad2deg(ref_rad))

        # 4) invert forward model: angle(deg) -> joint6(rad)
        j6_sol, ok = solve_joint6_for_angle(
            model, target_deg=ref_deg, j_min=-2.9, j_max=+2.9, x0=joint_pos[5]
        )

        # clamp to your safe window
        # j6_cmd = float(np.clip(j6_sol, J6_MIN, J6_MAX))

        # Optional: rate-limit joint changes (gentler motion)
        # max_step = np.deg2rad(1.0)  # 1 deg/iter
        # j6_cmd = float(np.clip(j6_cmd, joint_pos[5] - max_step, joint_pos[5] + max_step))

        # 5) send full 6D target with updated joint 6
        joint_pos[5] = j6_sol
        list_to_setp(setp, joint_pos, offset=6)
        con.send(setp)

        # 6) busy-bit wait
        while True:
            state = con.receive()
            con.send(watchdog)
            if not state.output_bit_registers0_to_31:
                break
            time.sleep(0.005)
        new_capture()
        image_path = "/home/jack/literate-code/focused_image.jpg"
        _, _, angle_deg = detect_red_points_and_angle(image_path)
        err_deg = ref_deg - angle_deg
        # 7) log
        print(f"t={t:5.2f}s  ref={ref_deg:6.2f}°  meas={angle_deg:6.2f}°  "
            f"err={err_deg:6.2f}°  j6={joint_pos[5]:.3f} rad  [{'ok' if ok else 'extr'}]")

        t_log.append(t)
        ref_deg_log.append(ref_deg)
        meas_deg_log.append(angle_deg)
        err_deg_log.append(err_deg)
        j6_rad_log.append(joint_pos[5])

        time.sleep(dt)
    # --- end NN-tracking section ---

    # proceed with next mode
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
    # ---------- Metrics ----------
    ref = np.asarray(ref_deg_log, dtype=float)
    meas = np.asarray(meas_deg_log, dtype=float)

    # Optional: ignore any NaNs that might slip in
    mask = np.isfinite(ref) & np.isfinite(meas)
    ref = ref[mask]
    meas = meas[mask]

    # MSE in degrees^2
    err_deg = ref - meas
    mse_deg2 = np.mean(err_deg**2)
    rmse_deg = np.sqrt(mse_deg2)
    mae_deg = np.mean(np.abs(err_deg))

    # Same in radians if you want
    ref_rad = np.deg2rad(ref)
    meas_rad = np.deg2rad(meas)
    err_rad = ref_rad - meas_rad
    mse_rad2 = np.mean(err_rad**2)
    rmse_rad = np.sqrt(mse_rad2)

    print(f"MSE = {mse_deg2:.4f} deg^2   (RMSE = {rmse_deg:.4f} deg,  MAE = {mae_deg:.4f} deg)")
    print(f"MSE = {mse_rad2:.6f} rad^2   (RMSE = {rmse_rad:.6f} rad)")

    # Save to file
    with open(RUN_DIR / "metrics.txt", "w") as f:
        f.write(f"MSE_deg2: {mse_deg2}\nRMSE_deg: {rmse_deg}\nMAE_deg: {mae_deg}\n")
        f.write(f"MSE_rad2: {mse_rad2}\nRMSE_rad: {rmse_rad}\n")

if __name__ == "__main__":
    main()
