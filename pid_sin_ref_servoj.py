import sys
import time
import logging
from new_cam import new_capture, detect_red_points_and_angle
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import csv, shutil
from vessel_trajecotry_plot import make_ref_tortuous
RUN_ROOT = Path("PID_control")
RUN_DIR = RUN_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR.mkdir(parents=True, exist_ok=True)
print(f"[LOG] Saving outputs to: {RUN_DIR}")
from pathlib import Path
import time

BASE_DIR = Path("runs_pid")   
BASE_DIR.mkdir(parents=True, exist_ok=True)

def make_run_dir():
    # timestamp-based to avoid clashes
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = BASE_DIR / f"run_{stamp}"
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    return run_dir

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
-0.39337331453432256, -1.989443918267721, -1.5559700727462769, -1.1809083384326478, 1.539546012878418, 1.6148388385772705]



def main():
    global RUN_DIR
    RUN_DIR = make_run_dir()
    print(f"[RUN] Logging to {RUN_DIR}")
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
    # A_deg      = 15.0         # sine amplitude (deg)
    # bias_deg   = 0.0         # bias/offset (deg)
    # freq_hz    = 0.02        # sine frequency (Hz) -> period ~12.5 s
    # phase_deg  = 0.0          # initial phase (deg)
    # duration_s = 60.0         # run time (s)
    A_deg, bias_deg, freq_hz, phase_deg, duration_s = 20.0, 0.0, 2.5, 0.0, .5
    A_rad     = np.deg2rad(A_deg)
    bias_rad  = np.deg2rad(bias_deg)
    phase_rad = np.deg2rad(phase_deg)

    # PID gains (operate in radians)
    # PID gains (operate in radians)
    Kp, Ki, Kd = 0.8, 0.01, 0.01 # start with Ki=0; add later if needed

    # dt = 1.0 / FREQUENCY
    dt = 0.005
    MAX_STEP = np.inf      # <= 0.5° change per 40ms cycle (tune up to 1–2° if stable)
    ALPHA = 0.7                     # meas low-pass (0=no filter, 1=heavy filter)

    joint_pos = JOINT_TARGET.copy()
    SIGN = -1.0
    J6_MIN = JOINT_TARGET[5] - np.pi/2
    J6_MAX = JOINT_TARGET[5] + np.pi/2

    t0 = time.time()
    err_i = 0.0
    prev_err_rad = None
    meas_filt = None
    next_tick = time.perf_counter()
    ref = make_ref_tortuous(A_rad=0.5, f_slow=2.4, f_fast=6.5, offset_rad=0.0, phase_rad=0.0)

    # while True:
    #     t = time.time() - t0
    #     if t > duration_s:
    #         break
    N = int(round(duration_s/dt))
    for k in range(N):
        t = k*dt
        # 1) measure angle
        new_capture()
        image_path = "/home/jack/literate-code/focused_image.jpg"
        _, _, angle_deg = detect_red_points_and_angle(image_path)
        angle_rad = np.deg2rad(angle_deg)

        # low-pass the measurement to avoid jitter spikes -> accel errors
        meas_filt = angle_rad if meas_filt is None else (1-ALPHA)*meas_filt + ALPHA*angle_rad

        # 2) reference
        ref_rad = bias_rad + A_rad * np.sin(2.0 * np.pi * freq_hz * t + phase_rad)
        # theta_ref, theta_refdot = ref(t)   # tuple: (angle, angular rate)

        # === this replaces your old line ===
        # ref_rad = theta_ref
        # 3) PID
        err_rad = ref_rad - meas_filt
        err_i  += err_rad * dt
        # err_i = np.clip(err_i, np.deg2rad(-90), np.deg2rad(90))  # anti-windup
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


    watchdog.input_int_register_0 = 3
    con.send(watchdog)
    print("Mode 3 sent — robot should move to Halt section now.")
    time.sleep(0.1)
    con.send_pause()
    con.disconnect()

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
    # -------- SAVE & COMPARE (no lead/lag) --------
    (RUN_DIR / "plots").mkdir(exist_ok=True)

    t_arr       = np.asarray(t_log, dtype=float)
    ref_now_deg = np.asarray(ref_deg_log, dtype=float)
    meas_deg    = np.asarray(meas_deg_log, dtype=float)

    # Raw synchronous error (NO shift)
    err_raw_deg = ref_now_deg - meas_deg

    # Metrics (deg)
    mse_deg2  = float(np.mean(err_raw_deg**2))
    rmse_deg  = float(np.sqrt(mse_deg2))
    mae_deg   = float(np.mean(np.abs(err_raw_deg)))

    # Also in radians (optional)
    ref_rad = np.deg2rad(ref_now_deg)
    meas_rad = np.deg2rad(meas_deg)
    err_rad  = ref_rad - meas_rad
    mse_rad2 = float(np.mean(err_rad**2))
    rmse_rad = float(np.sqrt(mse_rad2))

    print(f"[METRIC] RMSE (deg): {rmse_deg:.3f} | MAE (deg): {mae_deg:.3f}")

    # --- Plots: Reference vs Measured (no lag) ---
    plt.figure(figsize=(10, 6))
    plt.plot(t_arr, ref_now_deg, label="Reference (deg)", linewidth=2)
    plt.plot(t_arr, meas_deg,    label="Measured (deg)",  linewidth=1.5)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlabel("Time (s)"); plt.ylabel("Angle (deg)")
    plt.title("Reference vs Measured (synchronous)")
    plt.legend(); plt.tight_layout()
    plt.savefig(RUN_DIR / "plots" / "angles_ref_meas_sync.png", dpi=200); plt.close()

    # --- Error plot (no lag) ---
    plt.figure(figsize=(10, 4))
    plt.plot(t_arr, err_raw_deg, label="Error = Ref − Meas (deg)")
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlabel("Time (s)"); plt.ylabel("Error (deg)")
    plt.title(f"Tracking Error (RMSE = {rmse_deg:.2f}°)")
    plt.legend(); plt.tight_layout()
    plt.savefig(RUN_DIR / "plots" / "error_raw.png", dpi=200); plt.close()

    # --- Save scalars CSV for quick comparison ---
    with open(RUN_DIR / "scalars.csv", "w") as f:
        f.write("t,ref_now_deg,meas_deg,err_raw_deg\n")
        for i in range(len(t_arr)):
            f.write(f"{t_arr[i]:.6f},{ref_now_deg[i]:.6f},{meas_deg[i]:.6f},{err_raw_deg[i]:.6f}\n")
    print(f"[LOG] Wrote {RUN_DIR/'scalars.csv'}")

    # --- Save metrics CSV (RMSE etc.) ---
    with open(RUN_DIR / "metrics.csv", "w") as f:
        f.write("metric,value\n")
        f.write(f"RMSE_deg,{rmse_deg:.6f}\n")
        f.write(f"MAE_deg,{mae_deg:.6f}\n")
        f.write(f"MSE_deg2,{mse_deg2:.6f}\n")
        f.write(f"RMSE_rad,{rmse_rad:.8f}\n")
        f.write(f"MSE_rad2,{mse_rad2:.8f}\n")
    print(f"[LOG] Wrote {RUN_DIR/'metrics.csv'}")

    # --- NPZ for aggregator compatibility (no lead/lag) ---
    # Keep the key names your MPC aggregator expects; set aligned == raw (no shifting).
    np.savez(
        RUN_DIR / "pid_rollout_logs.npz",
        t=t_arr,
        ref_now_deg=ref_now_deg,
        meas_deg=meas_deg,
        err_raw_deg=err_raw_deg,
        err_aligned_deg=err_raw_deg,   # <- same as raw; still NO lag
        # Optional placeholders for parity (safe to ignore downstream)
        u0_deg_s=np.asarray(u_deg_log, dtype=float),
        psi_now_rad=np.full_like(t_arr, np.nan, dtype=float),
        psi_cmd_rad=np.full_like(t_arr, np.nan, dtype=float),
        qp_infeasible=np.zeros_like(t_arr, dtype=int),
        ref_seq_deg=np.array([], dtype=object),
        theta_pred_deg=np.array([], dtype=object),
        u_seq_deg_s=np.array([], dtype=object),
        psi_pred_rad=np.array([], dtype=object),
    )
    print(f"[LOG] Wrote {RUN_DIR/'pid_rollout_logs.npz'}")
if __name__ == "__main__":
    main()
