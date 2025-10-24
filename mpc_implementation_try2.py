# beam
# Joint State: [-0.3329866568194788, -2.0541001758971156, -1.0448529720306396, -1.6290365658202113, 1.5409388542175293, 0.4389771819114685]
# TCP Pose:  [0.8626541834323219, -0.4868659841674622, 0.6304371536197035, 1.206710674372489, -2.8637625212082796, 0.021032903808600848]
# robot
# Joint State: [-0.45086700121034795, -1.9217144451537074, -1.6537175178527832, -1.1486450296691437, 1.5386974811553955, 0.32281869649887085]
# TCP Pose:  [0.6863470071188493, -0.5297771147199332, 0.3988899776980005, 1.2068194033944717, -2.8637413293536538, 0.02102025865000871]
#beam 
#Joint State: [-0.2986090818988245, -2.1936494312682093, -0.5000160932540894, -2.0353552303709925, 1.5415232181549072, 0.8605273365974426]
#TCP Pose:  [0.8944494784532359, -0.46211378735359915, 0.8086953273783671, 0.6313943668787805, -3.046134496660519, 0.030694695737150387]


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
from neural_net import SimpleMLP
import torch
# --- before the loop, build the controller once ---
from mpc_controller import MPCController

# nn_theta_jacobian.py
import torch, numpy as np
log=True
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
def make_theta_J_from_model(model: torch.nn.Module):
    """Return (theta_fn, J_fn) from a PyTorch model mapping ψ(rad)->θ(deg)."""
    model.eval()
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    def theta_fn(psi_rad: float) -> float:
        with torch.no_grad():
            x = torch.tensor([[float(psi_rad)]], dtype=torch.float32, device=device)
            y_deg = model(x).item()           # θ in degrees
        return np.deg2rad(y_deg)              # θ in radians

    def J_fn(psi_rad: float) -> float:
        x = torch.tensor([[float(psi_rad)]], dtype=torch.float32, device=device, requires_grad=True)
        y_deg = model(x)                      # θ in degrees
        (dy_dpsi_deg_per_rad,) = torch.autograd.grad(y_deg, x, torch.ones_like(y_deg), retain_graph=False)
        # Convert deg/rad -> rad/rad:
        return float(dy_dpsi_deg_per_rad.item()) * (np.pi / 180.0)

    return theta_fn, J_fn


# Load your trained model (CPU is fine; GPU works too)
model = SimpleMLP()
model.load_state_dict(torch.load("simple_mlp.pt", map_location="cpu"))
model.eval()

theta_fn, J_fn = make_theta_J_from_model(model)
controller = MPCController(
    theta_fn=theta_fn,
    J_fn=J_fn,
    dt=0.2,
    Np=4,                 # you used 5 in the sim main
    w_th=10,
    w_u=1,
    theta_band_deg=np.inf,    # no band in your sim run (you had error_threshold=np.inf)
    eps_theta_deg=10.0,
    h_deg_for_radius=0.5,
    trust_region_deg=180.0,
    theta_max_deg=180.0,
    u_max_deg_s=180.0,
    j6_min_rad=-5,
    j6_max_rad=+5,
    rate_limit_deg=180.0
)


RUN_ROOT = Path("MPC_control")
RUN_DIR = RUN_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR.mkdir(parents=True, exist_ok=True)
print(f"[LOG] Saving outputs to: {RUN_DIR}")
(RUN_DIR / "plots").mkdir(exist_ok=True)

def list_to_setp(setp, lst, offset=0):
    """ Converts a list into RTDE input registers, allowing different sets with offsets. """
    for i in range(6):
        setp.__dict__[f"input_double_register_{i + offset}"] = lst[i]
    return setp

ROBOT_HOST = "192.168.56.101"
ROBOT_PORT = 30004
CONFIG_XML = "control_loop_configuration.xml"
FREQUENCY = 50  # use 125 if your controller prefers it

# input_double_registers 6..11 hold the joint target q[0..5]
JOINT_TARGET = [
-0.413648907338278,-2.00402083019399,-1.53428280353546,-1.18701540053401,1.53928232192993,1.13081705570221
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

    # --- handshake to "Run" mode (your code) ---
    # --- mode enter ---
    watchdog.input_int_register_0 = 2
    con.send(watchdog)
    state = con.receive()

    # --- holders ---
    prev_roll_times = None
    prev_roll_theta_deg = None
    prev_roll_ref_deg = None

    dt_log = []
    t_pre_log, t_post_log = [], []
    meas_pre_deg_log, meas_post_deg_log = [], []

    # per-tick controller/plant logs
    u0_deg_s_log = []
    psi_now_rad_log, psi_cmd_rad_log = [], []
    qp_status_log = []

    # diagnostic horizons
    ref_seq_deg_log = []        # list[(Np,)]
    theta_pred_deg_log = []     # list[(Np,)]
    u_seq_deg_s_log = []        # list[(Np,)]
    psi_pred_rad_log = []       # list[(Np,)]

    # k+1 (for display)
    pred_kp1_deg_log = []
    ref_kp1_deg_log  = []
    t_kp1_log        = []

    # causal, post-actuation metrics (aligned to previous rollout)
    ref_causal_post_deg_log = []
    err_causal_post_deg_log = []          # (ref_prev_aligned - meas_post)
    consistency_post_deg_log = []         # (meas_post - theta_hat_prev_aligned)

    # init ψ from robot
    joint_pos = np.array(state.actual_q, dtype=float)
    controller.set_initial_psi(joint_pos[5])

    # reference
    A_deg, bias_deg, freq_hz, phase_deg, duration_s = 15.0, 0.0, 0.02, 0.0, 60.0
    A_rad, bias_rad, phase_rad = np.deg2rad(A_deg), np.deg2rad(bias_deg), np.deg2rad(phase_deg)
    def ref_func(t): return bias_rad + A_rad * np.sin(2.0*np.pi*freq_hz*t + phase_rad)

    # timing
    t0 = time.time()
    last_tick = time.time()
    alpha_dt = 0.3
    dt_ema = None

    # vision
    image_path = "/home/jack/literate-code/focused_image.jpg"
    last_angle_pre_deg = 0.0

    while True:
        now = time.time()
        t_base = now - t0
        if t_base > duration_s:
            break

        # loop period and controller dt (smoothed for MPC)
        dt_k_raw = now - last_tick
        last_tick = now
        dt_k = float(np.clip(dt_k_raw, 0.10, 1.50))
        dt_ema = dt_k if dt_ema is None else (alpha_dt*dt_k + (1.0 - alpha_dt)*dt_ema)
        controller.set_dt(dt_ema)
        dt_log.append(dt_ema)

        # --- pre-actuation capture (state for control) ---
        new_capture()
        try:
            _, _, angle_pre_deg = detect_red_points_and_angle(image_path)
        except Exception:
            angle_pre_deg = last_angle_pre_deg
        last_angle_pre_deg = angle_pre_deg
        t_pre_log.append(t_base)
        meas_pre_deg_log.append(angle_pre_deg)

        # horizon planned at this tick (optionally include lookahead tau)
        tau = 1 * dt_ema
        ref_seq = np.array([ref_func(t_base + tau + (i+1)*dt_ema) for i in range(controller.Np)], float)

        # --- MPC solve using *pre* measurement ---
        j6_cmd_rad, info = controller.step_with_seq(ref_seq, np.deg2rad(angle_pre_deg))

        # unpack current rollout (this will become "previous" for *next* tick)
        u_seq    = np.asarray(info["u_seq_rad_s"], float)
        psi_pred = np.asarray(info["psi_pred_rad"], float)
        th_pred  = np.asarray(info["theta_pred_rad"], float)
        xref     = np.asarray(info["xref_seq_rad"], float)
        psi_k    = float(info["psi_now_rad"])

        # log horizons for this tick (diagnostics)
        ref_seq_deg_log.append(np.rad2deg(xref).astype(float))
        theta_pred_deg_log.append(np.rad2deg(th_pred).astype(float))
        u_seq_deg_s_log.append(np.rad2deg(u_seq).astype(float))
        psi_pred_rad_log.append(psi_pred.astype(float))

        # k+1 snapshot (for timeline plot)
        if th_pred.size > 0 and np.isfinite(th_pred[0]):
            pred_kp1_deg_log.append(float(np.degrees(th_pred[0])))
            ref_kp1_deg_log.append(float(np.degrees(xref[0])))
            t_kp1_log.append(t_base + dt_ema + tau)
        else:
            pred_kp1_deg_log.append(np.nan)
            ref_kp1_deg_log.append(np.nan)
            t_kp1_log.append(t_base + dt_ema + tau)

        # --- command robot (joint 6), then wait until move completes ---
        joint_pos[5] = j6_cmd_rad
        list_to_setp(setp, joint_pos, offset=6)
        con.send(setp)
        while True:
            state = con.receive()
            con.send(watchdog)
            if not state.output_bit_registers0_to_31:
                break
            time.sleep(0.005)

        # optional tiny settle to include exposure/servo lag
        time.sleep(0.02)

        # --- post-actuation capture (evaluation) ---
        t_post = time.time() - t0
        new_capture()
        try:
            _, _, angle_post_deg = detect_red_points_and_angle(image_path)
        except Exception:
            angle_post_deg = angle_pre_deg

        t_post_log.append(t_post)
        meas_post_deg_log.append(angle_post_deg)

        # --- causal alignment: compare post image to *previous* rollout ---
        if prev_roll_times is not None:
            t_eval = t_post
            t_first, t_last = prev_roll_times[0], prev_roll_times[-1]
            if t_eval <= t_first:
                theta_hat = float(prev_roll_theta_deg[0]); ref_aligned = float(prev_roll_ref_deg[0])
            elif t_eval >= t_last:
                theta_hat = float(prev_roll_theta_deg[-1]); ref_aligned = float(prev_roll_ref_deg[-1])
            else:
                theta_hat = float(np.interp(t_eval, prev_roll_times, prev_roll_theta_deg))
                ref_aligned = float(np.interp(t_eval, prev_roll_times, prev_roll_ref_deg))

            # causal, post-actuation metrics
            consistency_post = angle_post_deg - theta_hat
            err_causal_post  = ref_aligned - angle_post_deg

            consistency_post_deg_log.append(consistency_post)
            err_causal_post_deg_log.append(err_causal_post)
            ref_causal_post_deg_log.append(ref_aligned)
        else:
            consistency_post_deg_log.append(np.nan)
            err_causal_post_deg_log.append(np.nan)
            ref_causal_post_deg_log.append(np.nan)

        # --- per-tick scalar logs (same cadence as post capture) ---
        u0_deg_s_log.append(float(np.rad2deg(info['u0_rad_s'])))
        psi_now_rad_log.append(psi_k)
        psi_cmd_rad_log.append(float(info['psi_cmd_rad']))
        qp_status_log.append(1 if info.get('infeasible', False) else 0)

        # --- prepare "previous" rollout for the *next* tick alignment ---
        th_deg  = np.rad2deg(th_pred).astype(float)
        rx_deg  = np.rad2deg(xref).astype(float)
        prev_roll_times     = t_base + (np.arange(1, len(th_deg)+1) * float(controller.dt))
        prev_roll_theta_deg = th_deg
        prev_roll_ref_deg   = rx_deg

    # --- mode exit ---
    watchdog.input_int_register_0 = 3
    con.send(watchdog)
    time.sleep(0.1)
    con.send_pause()
    con.disconnect()

    # -------- pack logs (AFTER the loop) --------
    t_pre_arr    = np.array(t_pre_log, dtype=float)
    t_post_arr   = np.array(t_post_log, dtype=float)
    dt_arr       = np.array(dt_log, dtype=float)
    meas_pre_arr = np.array(meas_pre_deg_log, dtype=float)
    meas_post_arr= np.array(meas_post_deg_log, dtype=float)

    u0_deg_s_arr    = np.array(u0_deg_s_log, dtype=float)
    psi_now_rad_arr = np.array(psi_now_rad_log, dtype=float)
    psi_cmd_rad_arr = np.array(psi_cmd_rad_log, dtype=float)
    qp_infeas_arr   = np.array(qp_status_log, dtype=int)

    ref_causal_post_arr     = np.array(ref_causal_post_deg_log, dtype=float)
    err_causal_post_arr     = np.array(err_causal_post_deg_log, dtype=float)
    consistency_post_arr    = np.array(consistency_post_deg_log, dtype=float)

    pred_kp1_deg_arr = np.array(pred_kp1_deg_log, dtype=float)
    ref_kp1_deg_arr  = np.array(ref_kp1_deg_log, dtype=float)
    t_kp1_arr        = np.array(t_kp1_log, dtype=float)

    # horizons (ragged)
    ref_seq_deg_obj    = np.array(ref_seq_deg_log, dtype=object)
    theta_pred_deg_obj = np.array(theta_pred_deg_log, dtype=object)
    u_seq_deg_s_obj    = np.array(u_seq_deg_s_log, dtype=object)
    psi_pred_rad_obj   = np.array(psi_pred_rad_log, dtype=object)

    # metrics (causal, post-actuation)
    rmse_post_deg = float(np.sqrt(np.nanmean(err_causal_post_arr**2)))
    print(f"[METRIC] Causal RMSE (post-actuation): {rmse_post_deg:.3f} deg")

    if log:
        # CSV (post-actuation, causal)
        with open(RUN_DIR / "scalars_post_causal.csv", "w") as f:
            f.write("t_post,meas_post_deg,ref_causal_deg,err_causal_deg,consistency_deg,u0_deg_s,psi_now_rad,psi_cmd_rad,qp_infeasible\n")
            for i in range(len(t_post_arr)):
                f.write(f"{t_post_arr[i]:.6f},{meas_post_arr[i]:.6f},{ref_causal_post_arr[i]:.6f},"
                        f"{err_causal_post_arr[i]:.6f},{consistency_post_arr[i]:.6f},"
                        f"{u0_deg_s_arr[i]:.6f},{psi_now_rad_arr[i]:.6f},{psi_cmd_rad_arr[i]:.6f},{qp_infeas_arr[i]}\n")
        print(f"[LOG] Wrote {RUN_DIR/'scalars_post_causal.csv'}")

        # NPZ (everything)
        np.savez(
            RUN_DIR / "mpc_rollout_logs_post_causal.npz",
            dt=dt_arr,
            t_pre=t_pre_arr,
            t_post=t_post_arr,
            meas_pre_deg=meas_pre_arr,
            meas_post_deg=meas_post_arr,
            ref_causal_post_deg=ref_causal_post_arr,
            err_causal_post_deg=err_causal_post_arr,
            consistency_post_deg=consistency_post_arr,
            u0_deg_s=u0_deg_s_arr,
            psi_now_rad=psi_now_rad_arr,
            psi_cmd_rad=psi_cmd_rad_arr,
            qp_infeas=qp_infeas_arr,
            pred_kp1_deg=pred_kp1_deg_arr,
            ref_kp1_deg=ref_kp1_deg_arr,
            t_kp1=t_kp1_arr,
            ref_seq_deg=ref_seq_deg_obj,
            theta_pred_deg=theta_pred_deg_obj,
            u_seq_deg_s=u_seq_deg_s_obj,
            psi_pred_rad=psi_pred_rad_obj,
        )
        print(f"[LOG] Wrote {RUN_DIR/'mpc_rollout_logs_post_causal.npz'}")

        # -------- plots (post-actuation, causal) --------
        plt.figure(figsize=(10, 6))
        plt.plot(t_post_arr, meas_post_arr, label="Measured (post) [deg]", linewidth=1.6)
        plt.plot(t_post_arr, ref_causal_post_arr, label="Ref (causal, aligned) [deg]", linewidth=2.0)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.xlabel("Time (s)"); plt.ylabel("Angle (deg)")
        plt.title("Reference vs Measured — Post-Actuation, Causally Aligned")
        plt.legend(); plt.tight_layout()
        plt.savefig(RUN_DIR / "plots" / "angles_ref_meas_post_causal.png", dpi=200); plt.close()

        plt.figure(figsize=(10, 4))
        plt.plot(t_post_arr, err_causal_post_arr, label="Causal Error (post) [deg]")
        plt.axhline(0.0, linestyle="--", linewidth=1)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.xlabel("Time (s)"); plt.ylabel("Error (deg)")
        plt.title(f"Causal Tracking Error (post) — RMSE = {rmse_post_deg:.2f}°")
        plt.legend(); plt.tight_layout()
        plt.savefig(RUN_DIR / "plots" / "error_causal_post.png", dpi=200); plt.close()

        plt.figure(figsize=(10, 4))
        plt.plot(t_post_arr, consistency_post_arr, label="meas_post − θ̂_prev(aligned) [deg]")
        plt.axhline(0.0, linestyle="--", linewidth=1)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.xlabel("Time (s)"); plt.ylabel("Δ (deg)")
        plt.title("Prediction Consistency — Post-Actuation, Time-Aligned")
        plt.legend(); plt.tight_layout()
        plt.savefig(RUN_DIR / "plots" / "consistency_post.png", dpi=200); plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(t_kp1_arr, ref_kp1_deg_arr, label="R[k+1] (deg)", linewidth=2)
        plt.plot(t_kp1_arr, pred_kp1_deg_arr, label="θ̂[k+1] (deg)", linewidth=1.5)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.xlabel("Time (s)"); plt.ylabel("Angle (deg)")
        plt.title("MPC k+1 Prediction vs k+1 Reference (timestamps at k+1)")
        plt.legend(); plt.tight_layout()
        plt.savefig(RUN_DIR / "plots" / "k1_pred_vs_ref.png", dpi=200); plt.close()

        print(f"[LOG] Plots saved under {RUN_DIR/'plots'}")


if __name__ == "__main__":
    main()
