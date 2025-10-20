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
    Np=1,                 # you used 5 in the sim main
    w_th=3.0,
    w_u=1.0,
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
FREQUENCY = 125  # use 125 if your controller prefers it

# input_double_registers 6..11 hold the joint target q[0..5]
JOINT_TARGET = [
-0.421221081410543, -1.99183716396474, -1.55251479148865, -1.18077780426059, 1.53922581672669, 1.06926810741425]
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

    # --- handshake to "Run" mode (your code) ---
    watchdog.input_int_register_0 = 2
    con.send(watchdog)
    state = con.receive()
    t_log = []
    ref_seq_deg_log = []       # list of arrays (len Np) in degrees
    ref_now_deg_log = []       # scalar: first element of ref_seq
    meas_deg_log = []
    theta_pred_deg_log = []    # list of arrays (len Np)
    u0_deg_s_log = []
    u_seq_deg_s_log = []       # list of arrays (len Np)
    psi_now_rad_log = []
    psi_cmd_rad_log = []
    psi_pred_rad_log = []      # list of arrays (len Np)
    qp_status_log = []         # 0 ok, 1 infeasible
    # get current joints & init ψ (joint 6)
    joint_pos = np.array(state.actual_q, dtype=float)  # radians
    controller.set_initial_psi(joint_pos[5])

    # --- sine reference in radians, same as sim ---
    A_deg, bias_deg, freq_hz, phase_deg, duration_s = 15.0, 0.0, 0.02, 0.0, 60.0
    A_rad     = np.deg2rad(A_deg)
    bias_rad  = np.deg2rad(bias_deg)
    phase_rad = np.deg2rad(phase_deg)

    # RefStream-style helper (rad -> rad)
    def ref_func(t):
        return bias_rad + A_rad * np.sin(2.0*np.pi*freq_hz*t + phase_rad)

    # simple drop-in replacement for RefStream.sequence(...)
    def make_ref_seq(t0, Np, dt):
        # horizon lookahead at t+dt, t+2dt, ..., t+Np*dt  (matches your sim)
        return np.array([ref_func(t0 + (i+1)*dt) for i in range(Np)], dtype=float)

    last_angle_deg = 0.0

    # Adaptive timing init
    t0 = time.time()
    last_tick = time.time()             # fresh timestamp to avoid tiny first dt

    # Optional: smoothing for dt (EMA)
    alpha_dt = 0.3                      # 0=no smoothing, 1=use last value entirely
    dt_ema = None

    while True:
        now = time.time()
        t = now - t0
        if t > duration_s:
            break

        # Measure actual loop period and update controller dt
        dt_k_raw = now - last_tick
        last_tick = now

        # Clamp dt to sane bounds, e.g. 0.1s..1.5s (tune for your setup)
        dt_k = float(np.clip(dt_k_raw, 0.10, 1.50))

        # Optional: smooth
        if dt_ema is None:
            dt_ema = dt_k
        else:
            dt_ema = alpha_dt * dt_k + (1 - alpha_dt) * dt_ema

        controller.set_dt(dt_ema)   # rebuilds S matrix etc.

        # 1) measure θ (deg) from vision -> radians
        new_capture()
        image_path = "/home/jack/literate-code/focused_image.jpg"
        try:
            _, _, angle_deg = detect_red_points_and_angle(image_path)
        except Exception:
            angle_deg = last_angle_deg  # reuse last on failure
        theta_meas_rad = np.deg2rad(angle_deg)
        last_angle_deg = angle_deg
        tau_est = dt_ema                                 # <<< NEW (start with 1 cycle)
        ref_seq = np.array([ref_func(t + tau_est + (i+1)*dt_ema)
                            for i in range(controller.Np)], float)   # <<< CHANGED
        # 2) build Np-step reference sequence using the same dt used by the controller
        # ref_seq = np.array([ref_func(t + (i+1)*dt_ema) for i in range(controller.Np)], float)

        # 3) MPC step -> absolute ψ command for joint 6
        j6_cmd_rad, info = controller.step_with_seq(ref_seq, theta_meas_rad)

        # --- pretty rollout table ---
        Np = controller.Np
        u_seq      = np.asarray(info["u_seq_rad_s"], float)         # (Np,)
        psi_pred   = np.asarray(info["psi_pred_rad"], float)        # (Np,)
        theta_pred = np.asarray(info["theta_pred_rad"], float)      # (Np,)
        xref       = np.asarray(info["xref_seq_rad"], float)        # (Np,)
        psi_k = info["psi_now_rad"]
        h = np.deg2rad(0.2)
        J_auto = float(J_fn(psi_k))
        J_fd = float((theta_fn(psi_k + h) - theta_fn(psi_k - h)) / (2*h))
        print(f"[J] psi={np.degrees(psi_k):.2f}°  J_auto={J_auto:.4e} rad/rad  J_fd={J_fd:.4e} rad/rad")

        if np.all(np.isnan(theta_pred)):
            print(f"[Rollout] (infeasible at t={t:.3f}s) — no prediction.")
        else:
            dpsi_step = u_seq * dt_ema                              # per-step ψ change
            dpsi_cum  = controller.S_np @ u_seq                     # cumulative ψ change
            err       = theta_pred - xref

            psi_k = info["psi_now_rad"]
            header = (f"\n[Rollout] t={t:.3f}s  ψ_k={np.degrees(psi_k):+.2f}°  "
                    f"dt={dt_ema:.3f}s  max|Δψ|={np.max(np.abs(np.degrees(dpsi_cum))):.2f}°  "
                    f"max|θ̂−R|={np.max(np.abs(np.degrees(err))):.2f}°")
            print(header)
            print(" i |   u_i [deg/s] |  Δψ_i(step) [deg] |  Δψ_i(cum) [deg] |  ψ̂_i [deg] |  θ̂_i [deg] |  R_i [deg] | θ̂_i−R_i [deg]")
            for i in range(Np):
                print(f"{i+1:2d} | "
                    f"{np.degrees(u_seq[i]):+12.2f} | "
                    f"{np.degrees(dpsi_step[i]):+16.2f} | "
                    f"{np.degrees(dpsi_cum[i]):+15.2f} | "
                    f"{np.degrees(psi_pred[i]):+10.2f} | "
                    f"{np.degrees(theta_pred[i]):+10.2f} | "
                    f"{np.degrees(xref[i]):+9.2f} | "
                    f"{np.degrees(err[i]):+12.2f}")
        # One-step nonlinear vs linear check (right after step_with_seq)
        theta_pred_rad = np.asarray(info["theta_pred_rad"], float)  # (Np,)
        if theta_pred_rad.size > 0 and np.isfinite(theta_pred_rad[0]):
            theta_lin_next = np.rad2deg(theta_pred_rad[0])
        else:
            theta_lin_next = np.nan

        theta_nl_next = np.rad2deg(theta_fn(info['psi_now_rad'] + info['u0_rad_s'] * dt_ema))
        print(f"    θ_next_nl≈{theta_nl_next:5.2f}° vs θ_next_lin≈{theta_lin_next:5.2f}°")


        # 4) send full 6D target with updated joint 6
        joint_pos[5] = j6_cmd_rad
        list_to_setp(setp, joint_pos, offset=6)
        con.send(setp)

        # 5) busy-bit wait
        while True:
            state = con.receive()
            con.send(watchdog)
            if not state.output_bit_registers0_to_31:
                break
            time.sleep(0.005)

        # 6) logging
        ref_now_deg = float(np.rad2deg(ref_seq[0]))
        err_deg = ref_now_deg - angle_deg
        print(f"t={t:5.2f}s  dt={dt_ema:.3f}s  ref={ref_now_deg:6.2f}°  meas={angle_deg:6.2f}°  "
            f"err={err_deg:6.2f}°  j6={joint_pos[5]:.3f} rad  "
            f"u0={np.rad2deg(info['u0_rad_s']):6.2f}°/s  [{info['status']}]")

        # append scalars
        t_log.append(t)
        ref_now_deg_log.append(ref_now_deg)
        meas_deg_log.append(angle_deg)
        u0_deg_s_log.append(float(np.rad2deg(info['u0_rad_s'])))
        psi_now_rad_log.append(float(info['psi_now_rad']))
        psi_cmd_rad_log.append(float(info['psi_cmd_rad']))
        qp_status_log.append(1 if info['infeasible'] else 0)

        # append sequences
        ref_seq_deg_log.append(np.rad2deg(info['xref_seq_rad']).astype(float))
        theta_pred_deg_log.append(np.rad2deg(info['theta_pred_rad']).astype(float))
        u_seq_deg_s_log.append(np.rad2deg(info['u_seq_rad_s']).astype(float))
        psi_pred_rad_log.append(info['psi_pred_rad'].astype(float))


        tp = theta_pred_deg_log[-1]
        print(f"  θ_pred[+1..+{controller.Np}] ≈ {np.array2string(tp, precision=1, separator=', ')}")

    # proceed with next mode
    watchdog.input_int_register_0 = 3
    con.send(watchdog)
    print("Mode 3 sent — robot should move to Halt section now.")
    time.sleep(0.1)
    con.send_pause()
    con.disconnect()
    # -------- pack logs (AFTER the loop) --------
    t_arr            = np.array(t_log, dtype=float)
    ref_now_deg_arr  = np.array(ref_now_deg_log, dtype=float)
    meas_deg_arr     = np.array(meas_deg_log, dtype=float)
    u0_deg_s_arr     = np.array(u0_deg_s_log, dtype=float)
    psi_now_rad_arr  = np.array(psi_now_rad_log, dtype=float)
    psi_cmd_rad_arr  = np.array(psi_cmd_rad_log, dtype=float)
    qp_infeas_arr    = np.array(qp_status_log, dtype=int)
    err_deg_arr      = ref_now_deg_arr - meas_deg_arr

    # horizon (ragged per-step arrays -> object dtype for NPZ)
    ref_seq_deg_obj     = np.array(ref_seq_deg_log, dtype=object)
    theta_pred_deg_obj  = np.array(theta_pred_deg_log, dtype=object)
    u_seq_deg_s_obj     = np.array(u_seq_deg_s_log, dtype=object)
    psi_pred_rad_obj    = np.array(psi_pred_rad_log, dtype=object)
    if log == True:
        # -------- save files --------
        # 1) scalars CSV
        with open(RUN_DIR / "scalars.csv", "w") as f:
            f.write("t,ref_now_deg,meas_deg,err_deg,u0_deg_s,psi_now_rad,psi_cmd_rad,qp_infeasible\n")
            for i in range(len(t_arr)):
                f.write(f"{t_arr[i]:.6f},{ref_now_deg_arr[i]:.6f},{meas_deg_arr[i]:.6f},"
                        f"{err_deg_arr[i]:.6f},{u0_deg_s_arr[i]:.6f},"
                        f"{psi_now_rad_arr[i]:.6f},{psi_cmd_rad_arr[i]:.6f},{qp_infeas_arr[i]}\n")
        print(f"[LOG] Wrote {RUN_DIR/'scalars.csv'}")

        # 2) full NPZ (keeps horizon arrays intact)
        np.savez(
            RUN_DIR / "mpc_rollout_logs.npz",
            t=t_arr,
            ref_now_deg=ref_now_deg_arr,
            meas_deg=meas_deg_arr,
            err_deg=err_deg_arr,
            u0_deg_s=u0_deg_s_arr,
            psi_now_rad=psi_now_rad_arr,
            psi_cmd_rad=psi_cmd_rad_arr,
            qp_infeasible=qp_infeas_arr,
            ref_seq_deg=ref_seq_deg_obj,
            theta_pred_deg=theta_pred_deg_obj,
            u_seq_deg_s=u_seq_deg_s_obj,
            psi_pred_rad=psi_pred_rad_obj,
        )
        print(f"[LOG] Wrote {RUN_DIR/'mpc_rollout_logs.npz'}")

        # -------- plots (same style you had) --------
        plt.figure(figsize=(10, 6))
        plt.plot(t_arr, ref_now_deg_arr, label="Reference (deg)", linewidth=2)
        plt.plot(t_arr, meas_deg_arr,    label="Measured (deg)", linewidth=1.5)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.xlabel("Time (s)"); plt.ylabel("Angle (deg)")
        plt.title("Sine Tracking: Reference vs Measured"); plt.legend(); plt.tight_layout()
        plt.savefig(RUN_DIR / "plots" / "angles_ref_meas.png", dpi=200); plt.close()

        plt.figure(figsize=(10, 4))
        plt.plot(t_arr, err_deg_arr, label="Error (deg)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.xlabel("Time (s)"); plt.ylabel("Error (deg)")
        plt.title("Tracking Error"); plt.legend(); plt.tight_layout()
        plt.savefig(RUN_DIR / "plots" / "error.png", dpi=200); plt.close()

        plt.figure(figsize=(10, 4))
        plt.plot(t_arr, np.rad2deg(psi_now_rad_arr), label="ψ_now (deg)")
        plt.plot(t_arr, np.rad2deg(psi_cmd_rad_arr), label="ψ_cmd (deg)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.xlabel("Time (s)"); plt.ylabel("ψ (deg)")
        plt.title("Joint 6 Command"); plt.legend(); plt.tight_layout()
        plt.savefig(RUN_DIR / "plots" / "j6.png", dpi=200); plt.close()

        # --- overlay predicted θ rollouts for context ---
        plt.figure(figsize=(10, 6))
        plt.plot(t_arr, meas_deg_arr,    color="k", linewidth=1.0, label="θ_meas (deg)")
        plt.plot(t_arr, ref_now_deg_arr, color="C0", linewidth=1.2, label="θ_ref (deg)")
        stride = max(1, len(t_arr)//20)  # ~20 rollouts
        for i in range(0, len(t_arr), stride):
            tp = theta_pred_deg_log[i]               # (Np,) degrees
            th = t_arr[i] + (np.arange(1, len(tp)+1) * float(controller.dt))
            plt.plot(th, tp, alpha=0.30)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.xlabel("Time (s)"); plt.ylabel("Angle (deg)")
        plt.title("Predicted θ Rollouts (overlaid)"); plt.legend(); plt.tight_layout()
        plt.savefig(RUN_DIR / "plots" / "theta_rollouts.png", dpi=200); plt.close()

        print(f"[LOG] Plots saved under {RUN_DIR/'plots'}")


if __name__ == "__main__":
    main()
