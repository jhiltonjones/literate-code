import sys, time, logging, math
import numpy as np
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config
from new_cam import new_capture
import matplotlib.pyplot as plt
import cv2
start_point2 = [
    -0.4110644499408167, -1.8883592091002406, -1.699528455734253,
    -1.1374167960933228,  1.5391621589660645, -3.4619577566729944
]


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

ROBOT_HOST = "192.168.56.101"
ROBOT_PORT = 30004
CONFIG_XML = "control_loop_configuration.xml"
FREQUENCY = 500                 # 125 Hz RTDE is robust
CTRL_HZ = 10.0                  # PID loop rate (Hz)
CTRL_DT = 1.0 / CTRL_HZ

# Registers 6..11 carry joint targets
REG_BASE = 6

# === Calibration: angle(deg) -> wrist joint q6 (rad) ===
# Positive beam angle should rotate wrist joint in the correct sense.
# Start with a cautious small gain; tune on the rig.
K_Q_PER_DEG = math.radians(0.8)     # TODO: tune (rad joint) / (deg beam)
Q6_MIN = -2.8                        # joint hard clamp (rad)
Q6_MAX =  2.8

# === Reference bounds (deg) ===
REF_MIN_DEG = -15.0
REF_MAX_DEG =  60.0

# ----------------- PID -----------------
class PID:
    def __init__(self, Kp=1.2, Ki=0.1, Kd=0.2, dt=CTRL_DT, i_clamp=20.0, u_min=-0.5, u_max=0.5, d_tau=0.1):
        """
        u_min/u_max in *deg per step equivalent* (we convert to joint later).
        d_tau: derivative low-pass time constant (s)
        """
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.dt = dt
        self.int = 0.0
        self.i_clamp = i_clamp
        self.u_min, self.u_max = u_min, u_max
        self.prev_meas = None
        self.d_alpha = dt / (d_tau + dt)  # 1st-order LPF coeff
        self.d_est = 0.0

    def step(self, ref_deg, meas_deg):
        e = ref_deg - meas_deg

        # Proportional
        P = self.Kp * e

        # Integral with clamp (anti-windup)
        self.int += self.Ki * e * self.dt
        self.int = max(-self.i_clamp, min(self.int, self.i_clamp))

        # Derivative on measurement (robust to ref steps) with LPF
        if self.prev_meas is None:
            dm = 0.0
        else:
            dm = (meas_deg - self.prev_meas) / self.dt
        self.prev_meas = meas_deg
        self.d_est = (1 - self.d_alpha) * self.d_est + self.d_alpha * dm
        D = -self.Kd * self.d_est

        u = P + self.int + D
        # Clamp output
        u = max(self.u_min, min(u, self.u_max))
        return u  # u is a small angle correction (deg)
# --- add inside class PID ---
    def step_dbg(self, ref_deg, meas_deg):
        """
        Like step(), but also returns (u_deg, err_deg, P_deg, I_deg, D_deg).
        All values are in degrees units (u is deg-equivalent before mapping to joint).
        """
        e = ref_deg - meas_deg

        # P
        P = self.Kp * e

        # I with anti-windup
        self.int += self.Ki * e * self.dt
        self.int = max(-self.i_clamp, min(self.int, self.i_clamp))
        I = self.int

        # D on measurement with LPF
        if self.prev_meas is None:
            dm = 0.0
        else:
            dm = (meas_deg - self.prev_meas) / self.dt
        self.prev_meas = meas_deg
        self.d_est = (1 - self.d_alpha) * self.d_est + self.d_alpha * dm
        D = -self.Kd * self.d_est

        u = P + I + D
        u = max(self.u_min, min(u, self.u_max))
        return u, e, P, I, D

# ------------- Trajectory -------------
def ref_generator(kind="step60", t=0.0):
    """
    Returns reference angle in deg in [REF_MIN_DEG, REF_MAX_DEG].
    Pick one: "stair", "sine", "step60", "triangle".
    """
    if kind == "sine":
        # mid at 22.5 deg, amplitude 37.5 deg (stays within -15..60)
        mid = 0.5*(REF_MIN_DEG + REF_MAX_DEG)
        amp = 0.5*(REF_MAX_DEG - REF_MIN_DEG)
        return mid + amp * math.sin(2*math.pi*0.03*t)  # 0.03 Hz
    elif kind == "step60":
        return 20.0
    elif kind == "triangle":
        # 60s period triangle between min & max
        period = 60.0
        ph = (t % period) / period
        r = 4*ph if ph < 0.25 else (2 - 4*ph if ph < 0.75 else (4*ph - 4))
        # r in [-1,1]
        return 0.5*(REF_MIN_DEG + REF_MAX_DEG) + 0.5*(REF_MAX_DEG - REF_MIN_DEG)*r
    else:  # "stair"
        # 10s steps cycling through a few levels
        levels = [-10, 0, 15, 30, 45, 60, 30, 0]
        idx = int(t // 10) % len(levels)
        return float(levels[idx])

def clamp(v, vmin, vmax): return vmin if v < vmin else (vmax if v > vmax else v)

# ------------- RTDE Helpers -------------
def write_q_target(setp, q):
    for i in range(6):
        setattr(setp, f"input_double_register_{REG_BASE + i}", float(q[i]))

def wait_ready(con):
    while True:
        st = con.receive()
        if st is None: raise RuntimeError("RTDE timeout (wait_ready)")
        if st.output_bit_registers0_to_31 & 0x1: return st

def wait_done(con):
    while True:
        st = con.receive()
        if st is None: raise RuntimeError("RTDE timeout (wait_done)")
        if (st.output_bit_registers0_to_31 & 0x1) == 0: return st

# ------------- MAIN CTRL LOOP -------------
def run_pid():
    logging.getLogger().setLevel(logging.INFO)

    conf = rtde_config.ConfigFile(CONFIG_XML)
    state_names, state_types = conf.get_recipe("state")
    setp_names, setp_types = conf.get_recipe("setp")
    watchdog_names, watchdog_types = conf.get_recipe("watchdog")

    con = rtde.RTDE(ROBOT_HOST, ROBOT_PORT)
    while con.connect() != 0:
        print("RTDE connect failed, retrying...")
        time.sleep(0.5)
    print("Connected.")

    try:
        con.get_controller_version()
        con.send_output_setup(state_names, state_types, FREQUENCY)
        setp = con.send_input_setup(setp_names, setp_types)
        watchdog = con.send_input_setup(watchdog_names, watchdog_types)

        for i in range(24):
            setattr(setp, f"input_double_register_{i}", 0.0)
        setp.input_bit_registers0_to_31 = 0
        watchdog.input_int_register_0 = 0

        if not con.send_start():
            print("Failed to send_start()")
            sys.exit(1)

        # --- Arm once on the pendant ---
        print("PID Mode (3): press CONTINUE on Polyscope to arm streaming...")
        ready_state = wait_ready(con)
        # --- Move to start_point2 using Mode 1 (single movej) ---
        print("Moving to start_point2...")
        write_q_target(setp, start_point2)
        watchdog.input_int_register_0 = 1  # Mode 1
        con.send(setp)
        con.send(watchdog)
        wait_done(con)  # wait until move completes
        print("Reached start_point2.")

        # --- Arm again for PID streaming (Mode 3) ---
        print("Now press CONTINUE on Polyscope to arm PID streaming (Mode 3)...")
        ready_state = wait_ready(con)  # bit0=1 again for streaming
        q_now = list(ready_state.actual_q)  # start PID from the reached pose


        pid = PID(Kp=1.8, Ki=0.15, Kd=0.35, dt=CTRL_DT,
                  i_clamp=15.0, u_min=-2.0, u_max=2.0, d_tau=0.25)

        t0 = time.monotonic()
        next_tick = t0

        while True:
            # 1) Get measurement
            new_capture()
            image_path = "/home/jack/literate-code/focused_image.jpg"
            pt1, pt2, angle_deg = detect_red_points_and_angle(image_path)
            # If detection fails, you should handle it; example:
            if angle_deg is None or not np.isfinite(angle_deg):
                print("No angle detected; holding position")
                angle_deg = angle_deg if angle_deg is not None else 0.0

            # 2) Reference within [-15, 60]
            t = time.monotonic() - t0
            ref_deg = clamp(ref_generator(kind="step60", t=t), REF_MIN_DEG, REF_MAX_DEG)

            # 3) PID (in deg)
            u_deg, e_deg, P_deg, I_deg, D_deg = pid.step_dbg(ref_deg, angle_deg)


            # 4) Map to wrist joint (rad)
            dq6 = K_Q_PER_DEG * u_deg
            q_now[5] = clamp(q_now[5] + dq6, Q6_MIN, Q6_MAX)
            # Print one concise line per tick
            print(f"[{t:6.2f}s] ref={ref_deg:7.2f}°, meas={angle_deg:7.2f}°, "
                f"err={e_deg:7.2f}°, P={P_deg:7.2f}, I={I_deg:7.2f}, D={D_deg:7.2f}, "
                f"u={u_deg:6.2f}°  -> dq6={dq6:+.5f} rad, q6={q_now[5]:+.5f} rad")



            # 5) Send joint target via Mode 3 (PID streaming)
            write_q_target(setp, q_now)
            watchdog.input_int_register_0 = 3
            con.send(setp)
            con.send(watchdog)

            # 6) Handshake: one small move done
            wait_done(con)   # Polyscope clears bit0
            # Polyscope auto-arms bit0 True again (Mode 3), so no wait_ready here.

            # 7) Keep loop timing steady
            next_tick += CTRL_DT
            sleep_left = next_tick - time.monotonic()
            if sleep_left > 0:
                time.sleep(sleep_left)
            else:
                # running behind—reset schedule to avoid drift
                next_tick = time.monotonic()

    finally:
        try: con.send_pause()
        except Exception: pass
        try: con.disconnect()
        except Exception: pass
        print("RTDE session closed.")

if __name__ == "__main__":
    run_pid()
