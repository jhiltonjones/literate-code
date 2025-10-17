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
RUN_ROOT = Path("PID_control")
RUN_DIR = RUN_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR.mkdir(parents=True, exist_ok=True)
print(f"[LOG] Saving outputs to: {RUN_DIR}")

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
-0.45087892213930303, -1.9217030010619105, -1.6536900997161865, -1.1486326617053528, 1.5386854410171509, -1.8397100607501429
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

    watchdog.input_int_register_0 = 2
    con.send(watchdog)
    state = con.receive()
    # logs
    t_log, ref_deg_log, meas_deg_log = [], [], []
    err_deg_log, u_deg_log, j6_rad_log = [], [], []

    # --- PID tracking of a sine reference (all radians internally) ---
    # ref(t) = bias + A * sin(2π f t + φ)
    A_deg      = 15.0         # sine amplitude (deg)
    bias_deg   = 0.0         # bias/offset (deg)
    freq_hz    = 0.02        # sine frequency (Hz) -> period ~12.5 s
    phase_deg  = 0.0          # initial phase (deg)
    duration_s = 60.0         # run time (s)

    A_rad     = np.deg2rad(A_deg)
    bias_rad  = np.deg2rad(bias_deg)
    phase_rad = np.deg2rad(phase_deg)

    # PID gains (operate in radians)
    # PID gains (operate in radians)
    Kp, Ki, Kd = 0.5, 0.00, 0.08  # start with Ki=0; add later if needed

    dt = 1.0 / FREQUENCY

    Kp, Ki, Kd = 0.5, 0.0, 0.05      # slightly softer D
    MAX_STEP = np.deg2rad(10)       # <= 0.5° change per 40ms cycle (tune up to 1–2° if stable)
    ALPHA = 0.7                     # meas low-pass (0=no filter, 1=heavy filter)

    joint_pos = JOINT_TARGET.copy()
    SIGN = 1.0
    J6_MIN = JOINT_TARGET[5] - np.pi/2
    J6_MAX = JOINT_TARGET[5] + np.pi/2

    t0 = time.time()
    err_i = 0.0
    prev_err_rad = None
    meas_filt = None
    next_tick = time.perf_counter()

    while True:
        t = time.time() - t0
        if t > duration_s:
            break

        # 1) measure angle
        new_capture()
        image_path = "/home/jack/literate-code/focused_image.jpg"
        _, _, angle_deg = detect_red_points_and_angle(image_path)
        angle_rad = np.deg2rad(angle_deg)

        # low-pass the measurement to avoid jitter spikes -> accel errors
        meas_filt = angle_rad if meas_filt is None else (1-ALPHA)*meas_filt + ALPHA*angle_rad

        # 2) reference
        ref_rad = bias_rad + A_rad * np.sin(2.0 * np.pi * freq_hz * t + phase_rad)

        # 3) PID
        err_rad = ref_rad - meas_filt
        err_i  += err_rad * dt
        err_i = np.clip(err_i, np.deg2rad(-30), np.deg2rad(30))  # anti-windup
        derr   = 0.0 if prev_err_rad is None else (err_rad - prev_err_rad) / dt
        u_rad  = Kp * err_rad + Ki * err_i + Kd * derr
        prev_err_rad = err_rad

        # 4) APPLY with per-cycle rate limit (prevents accel violations)
        step = np.clip(SIGN * u_rad, -MAX_STEP, MAX_STEP)
        joint_pos[5] = float(np.clip(joint_pos[5] + step, J6_MIN, J6_MAX))

        # write regs 6..11
        list_to_setp(setp, joint_pos, offset=6)
        if "input_bit_registers0_to_31" in setp_names:
            setp.input_bit_registers0_to_31 = 0
        con.send(setp)

        # 5) keepalive + precise pacing (ONLY one sleep here)
        _ = con.receive()
        con.send(watchdog)
        next_tick += dt
        remaining = next_tick - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)

        # 6) log
        print(f"t={t:5.2f}s  ref={np.rad2deg(ref_rad):6.2f}°  "
            f"meas={np.rad2deg(meas_filt):6.2f}°  err={np.rad2deg(err_rad):6.2f}°  "
            f"u={np.rad2deg(u_rad):5.2f}°  j6={joint_pos[5]:.3f} rad")
        t_log.append(t)
        ref_deg_log.append(np.rad2deg(ref_rad))
        meas_deg_log.append(np.rad2deg(meas_filt))
        err_deg_log.append(np.rad2deg(err_rad))
        u_deg_log.append(np.rad2deg(u_rad))
        j6_rad_log.append(joint_pos[5])


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
    plt.plot(t_log, u_deg_log, label="Control step u (deg/iter)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlabel("Time (s)"); plt.ylabel("u (deg)")
    plt.title("Control Effort per Iteration"); plt.legend(); plt.tight_layout()
    plt.savefig(RUN_DIR / "plots" / "u.png", dpi=200)

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
