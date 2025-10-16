import sys
import time
import logging
from new_cam import new_capture
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config
import matplotlib.pyplot as plt
import cv2
import numpy as np

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
FREQUENCY = 125  # use 125 if your controller prefers it

# input_double_registers 6..11 hold the joint target q[0..5]
JOINT_TARGET = [
-0.4508908430682581, -1.921654840508932, -1.6536662578582764, -1.148660497074463, 1.5386666059494019, -5.430761162434713
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

    # --- Simple PID in *radians* to drive measured angle toward REF_ANGLE_DEG ---
    REF_ANGLE_DEG = 20.0
    REF_ANGLE_RAD = np.deg2rad(REF_ANGLE_DEG)

    # PID gains (all in rad-domain)
    Kp, Ki, Kd = 1.0, 0.00, 0.05          # tune as needed (small & stable to start)
    i_clamp_rad = np.deg2rad(5.0)          # integral clamp (rad)
    tol_rad = np.deg2rad(3.0)              # stop when |error| < 1 deg
    dt = 0.2                               # seconds between measurements

    joint_pos = JOINT_TARGET.copy()
    err_i = 0.0
    prev_err_rad = None

    # If the camera angle increases when joint6 increases, keep +1; flip to -1 if opposite.
    SIGN = +1.0

    # joint 6 limits (rad) – adjust if your robot differs
    # J6_MIN, J6_MAX = -2.9, 2.9
    J6_MIN = JOINT_TARGET[5] -np.pi/2
    J6_MAX = JOINT_TARGET[5] +np.pi/2
    for k in range(5):
        new_capture()
        image_path = "/home/jack/literate-code/focused_image.jpg"
        _, _, angle_deg = detect_red_points_and_angle(image_path)
        angle_rad = np.deg2rad(angle_deg)

        # error in radians
        err_rad = REF_ANGLE_RAD - angle_rad

        # PID (all radians)
        err_i = np.clip(err_i + err_rad * dt, -i_clamp_rad, i_clamp_rad)
        derr = 0.0 if prev_err_rad is None else (err_rad - prev_err_rad) / dt
        u_rad = Kp * err_rad + Ki * err_i + Kd * derr
        prev_err_rad = err_rad

        # apply to joint 6 (respect coupling sign)
        j6_new = np.clip(joint_pos[5] - SIGN * u_rad, J6_MIN, J6_MAX)
        joint_pos[5] = float(j6_new)

        list_to_setp(setp, joint_pos, offset=6)
        con.send(setp)

        # wait for robot "busy bit" to drop
        while True:
            state = con.receive()
            con.send(watchdog)
            if not state.output_bit_registers0_to_31:
                break
            time.sleep(0.005)

        print(f"[{k:02d}] angle={angle_deg:6.2f}°, err={np.rad2deg(err_rad):6.2f}°, "
            f"u={np.rad2deg(u_rad):5.2f}°, j6={joint_pos[5]:.3f} rad")

        if abs(err_rad) <= tol_rad:
            print("PID: within tolerance.")
            break
        time.sleep(dt)
    # --- end PID block ---



    # proceed with next mode
    watchdog.input_int_register_0 = 3
    con.send(watchdog)
    print("Mode 3 sent — robot should move to Halt section now.")
    time.sleep(0.1)
    con.send_pause()
    con.disconnect()


if __name__ == "__main__":
    main()
