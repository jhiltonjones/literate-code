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
from new_cam import new_capture, detect_red_points_and_angle
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
import torch
from vessel_trajecotry_plot import make_ref_tortuous
from mpc_controller_lti import MPC_controller_time_invariant
import torch, numpy as np
log=True

def make_theta_J_from_model(model: torch.nn.Module):
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
        return float(dy_dpsi_deg_per_rad.item()) * (np.pi / 180.0)

    return theta_fn, J_fn

model = SimpleMLP()
model.load_state_dict(torch.load("simple_mlp.pt", map_location="cpu"))
model.eval()

theta_fn, J_fn = make_theta_J_from_model(model)
controller = MPC_controller_time_invariant(
    J_fn=J_fn,
    dt=0.0,
    Np=6,                 
    w_th=250,
    w_u=0.005,
    theta_band_deg=np.inf,    
    eps_theta_deg=15.0,
    h_deg_for_radius=10.0,
    trust_region_deg=180.0,
    theta_max_deg=np.inf,
    u_max_deg_s=np.inf,
    rate_limit_deg=np.inf,
    use_stabilization=False
)


RUN_ROOT = Path("MPC_control")
RUN_DIR = RUN_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR.mkdir(parents=True, exist_ok=True)
print(f"[LOG] Saving outputs to: {RUN_DIR}")
(RUN_DIR / "plots").mkdir(exist_ok=True)
from pathlib import Path
import time

BASE_DIR = Path("runs_ltv2")       # or wherever you like
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
-0.3828681151019495, -2.021482606927389, -1.508305549621582, -1.1969867509654541, 1.539742350578308, 2.3771610260009766]



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

    # --- handshake to "Run" mode (your code) ---
    watchdog.input_int_register_0 = 2
    con.send(watchdog)
    state = con.receive()
    prev_theta_pred0 = None          # θ̂ for k+1 from previous cycle (rad)
    prev_xref0       = None          # R used for k+1 last cycle (rad)
    prev_stamp       = None
    dt_log = []
    err_raw_deg_log        = []
    err_aligned_deg_log    = []
    err_consistency_deg_log= []
    ref_aligned_deg_log    = []
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
    qp_status_log = []  
    pred_kp1_deg_log = []
    ref_kp1_deg_log  = []

    # get current joints & init ψ (joint 6)
    joint_pos = np.array(state.actual_q, dtype=float)  # radians
    controller.set_intial_psi(JOINT_TARGET[5])

    # --- sine reference in radians, same as sim ---
    A_deg, bias_deg, freq_hz, phase_deg, duration_s = 20.0, 0.0, 2.5, 0.0, .05
    # A_deg, bias_deg, freq_hz, phase_deg, duration_s = 15.0, 0.0, 0.02, 0.0, 60.0
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
    dt = 0.005
    dt_nom = 0.005
    controller.set_dt(dt_nom)              # build S_np etc. ONCE
    t_sim = 0.0
    N = int(round(duration_s/dt))
    ref = make_ref_tortuous(A_rad=0.2, f_slow=0.4, f_fast=1.5, offset_rad=0.0, phase_rad=0.0)

    # for k in range(N):
    #     t = k*dt
    #     # ref_seq = np.array([
    #     #     bias_rad + A_rad*np.sin(2*np.pi*freq_hz*(t_sim + (i+1)*dt_nom) + phase_rad)
    #     #     for i in range(controller.Np)
    #     # ], dtype=float)
    #     ref_seq = np.array([make_ref_tortuous(t_sim + (i+1)*dt_nom) for i in range(controller.Np)], dtype=float)
    #     dt_log.append(dt)
    for k in range(N):
        t = k * dt
        ref_seq = np.array([
            bias_rad + A_rad*np.sin(2*np.pi*freq_hz*(t_sim + (i+1)*dt_nom) + phase_rad)
            for i in range(controller.Np)
        ], dtype=float)
        # Horizon times (same logic as before: t_sim + (i+1)*dt_nom)
        # t_hzn = t_sim + dt_nom * np.arange(1, controller.Np + 2)

        # # Evaluate reference over the horizon
        # theta_hzn, theta_dot_hzn = ref(t_hzn)   # each is shape (Np,)

        # # ref_seq = np.array(theta_hzn, dtype=float)
        # ref_seq = np.array(theta_hzn[1:], dtype=float)
        controller.set_dt(dt)   # rebuilds S matrix etc.
        dt_log.append(dt)
        new_capture()
        image_path = "/home/jack/literate-code/focused_image.jpg"
        try:
            _, _, angle_deg = detect_red_points_and_angle(image_path)
        except Exception:
            angle_deg = last_angle_deg 
        if prev_theta_pred0 is not None:
            meas_vs_pred_next_deg = angle_deg - np.degrees(prev_theta_pred0)
            err_consistency_deg = meas_vs_pred_next_deg
        else:
            err_consistency_deg = np.nan

        if prev_xref0 is not None:
            ref_aligned_deg = float(np.rad2deg(prev_xref0))
            err_aligned_deg = ref_aligned_deg - angle_deg
        else:
            ref_aligned_deg = np.nan
            err_aligned_deg = np.nan           
        if prev_theta_pred0 is not None:
            print(f"[consistency] meas(t)={angle_deg:6.2f}°, prev θ̂₊₁={np.degrees(prev_theta_pred0):6.2f}°, "
                f"Δ={meas_vs_pred_next_deg:+5.2f}°  (Δt≈{t - prev_stamp:.2f}s)")
        t_sim += dt_nom
        ref_now_deg = float(np.rad2deg(ref_seq[0]))
        
        err_raw_deg = ref_now_deg - angle_deg

        t_log.append(t)
        ref_now_deg_log.append(ref_now_deg)          
        ref_aligned_deg_log.append(ref_aligned_deg) 
        meas_deg_log.append(angle_deg)
        err_aligned_deg_log.append(err_aligned_deg)
        err_consistency_deg_log.append(err_consistency_deg)

        # --- MPC step ---
        j6_cmd_rad, info = controller.step_with_seq(ref_seq, np.deg2rad(angle_deg))
        pk1 = info.get("theta_pred_rad")
        rk1 = info.get("xref_seq_rad")
        
        if pk1 is not None and len(pk1) > 0 and np.isfinite(pk1[0]):
            pred_kp1_deg_log.append(float(np.degrees(pk1[0])))
        else:
            pred_kp1_deg_log.append(np.nan)
        if rk1 is not None and len(rk1) > 0:
            ref_kp1_deg_log.append(float(np.degrees(rk1[0])))
        else:
            ref_kp1_deg_log.append(np.nan)

        theta_pred = np.asarray(info["theta_pred_rad"], float)  # (Np,)
        xref       = np.asarray(info["xref_seq_rad"], float)    # (Np,)


        # 2) Aligned tracking error: compare today's measurement to *yesterday’s planned* R (xref₊₁ from last cycle)
        if prev_xref0 is not None:
            ref_aligned_deg = float(np.rad2deg(prev_xref0))
            err_aligned_deg = ref_aligned_deg - angle_deg
            print(f"[aligned]  ref_prev₊₁={ref_aligned_deg:6.2f}°  meas={angle_deg:6.2f}°  "
                f"err_aligned={err_aligned_deg:+6.2f}°")
        else:
            ref_aligned_deg = np.nan
            err_aligned_deg = np.nan

        if prev_theta_pred0 is not None:
            err_consistency_deg = angle_deg - float(np.rad2deg(prev_theta_pred0))
        else:
            err_consistency_deg = np.nan

        prev_theta_pred0 = theta_pred[0]  
        prev_xref0       = xref[0]         
        prev_stamp       = t

        print(f"t={t:5.2f}s  dt={dt:.3f}s  ref={ref_now_deg:6.2f}°  meas={angle_deg:6.2f}°  "
            f"err_raw={err_raw_deg:+6.2f}°  err_aligned={err_aligned_deg:+6.2f}°  "
            f"j6={j6_cmd_rad:.3f} rad  u0={np.rad2deg(info['u0_rad_s']):6.2f}°/s  [{info['status']}]")
        psi_now = float(info['psi_now_rad'])
        u0 = float(info['u0_rad_s'])
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
            dpsi_step = u_seq * dt                             # per-step ψ change
            dpsi_cum  = controller.S_np @ u_seq                     # cumulative ψ change
            err       = theta_pred - xref

            psi_k = info["psi_now_rad"]
            header = (f"\n[Rollout] t={t:.3f}s  ψ_k={np.degrees(psi_k):+.2f}°  "
                    f"dt={dt:.3f}s  max|Δψ|={np.max(np.abs(np.degrees(dpsi_cum))):.2f}°  "
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
        theta_pred_rad = np.asarray(info["theta_pred_rad"], float)  # (Np,)
        theta_pred_rad = np.asarray(info["theta_pred_rad"], float)

        if theta_pred_rad.size > 0 and np.isfinite(theta_pred_rad[0]):
            theta_lin_next = np.rad2deg(theta_pred_rad[0])
        else:
            theta_lin_next = np.nan

        theta_nl_next = np.rad2deg(theta_fn(info['psi_now_rad'] + info['u0_rad_s']*dt))
        print(f"PSI NOW {info['psi_now_rad']}")
        print(f"U NOW {info['u0_rad_s']}")
        
        print(f"    θ_next_nl≈{theta_nl_next:5.2f}° vs θ_next_lin≈{theta_lin_next:5.2f}°")
        th_pred = np.asarray(info["theta_pred_rad"], float)
        # th_pred = th_pred.copy()
        # if th_pred.size > 0:
        #     th_pred[0] = theta_hat_k1

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

        # new_capture()
        # image_path = "/home/jack/literate-code/focused_image.jpg"

        # _, _, angle_deg_jack = detect_red_points_and_angle(image_path)

        # error_jack = ref_now_deg - angle_deg_jack

        # print(f"ERROR = {error_jack}")
        # print(f"JACK CON={angle_deg_jack:6.2f}°, prev θ̂₊₁={np.degrees(prev_theta_pred0):6.2f}°")
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
        print("----------------------------------------------------------------------------------------")
    # proceed with next mode
    watchdog.input_int_register_0 = 3
    con.send(watchdog)
    print("Mode 3 sent — robot should move to Halt section now.")
    time.sleep(0.1)
    con.send_pause()
    con.disconnect()
     # -------- pack logs (AFTER the loop) --------

    # Make lengths consistent
    # N = min(
    #     len(t_log), len(ref_now_deg_log), len(meas_deg_log),
    #     len(u0_deg_s_log), len(psi_now_rad_log), len(psi_cmd_rad_log),
    #     len(qp_status_log), len(dt_log),
    #     len(ref_aligned_deg_log), len(err_aligned_deg_log),
    # )

    N_core = min(len(t_log), len(ref_now_deg_log), len(meas_deg_log))
    if N_core == 0:
        print("[WARN] No samples collected; skipping metrics/plots.")
        return

    # Core arrays (use N_core)
    t_arr           = np.array(t_log[:N_core], dtype=float)
    ref_now_deg_arr = np.array(ref_now_deg_log[:N_core], dtype=float)
    meas_deg_arr    = np.array(meas_deg_log[:N_core], dtype=float)

    # Optional streams (clip to N_core)
    u0_deg_s_arr    = np.array(u0_deg_s_log[:N_core], dtype=float) if len(u0_deg_s_log)    >= N_core else np.full(N_core, np.nan)
    psi_now_rad_arr = np.array(psi_now_rad_log[:N_core], dtype=float) if len(psi_now_rad_log) >= N_core else np.full(N_core, np.nan)
    psi_cmd_rad_arr = np.array(psi_cmd_rad_log[:N_core], dtype=float) if len(psi_cmd_rad_log) >= N_core else np.full(N_core, np.nan)
    qp_infeas_arr   = np.array(qp_status_log[:N_core], dtype=int)   if len(qp_status_log)   >= N_core else np.zeros(N_core, dtype=int)
    dt_arr          = np.array(dt_log[:N_core], dtype=float)        if len(dt_log)          >= N_core else np.full(N_core, dt)

    # Aligned signals (xref[k-1] vs meas[k])
    ref_aligned_deg_arr = np.array(ref_aligned_deg_log[:N_core], dtype=float) if len(ref_aligned_deg_log) >= N_core else np.full(N_core, np.nan)
    err_aligned_deg_arr = np.array(err_aligned_deg_log[:N_core], dtype=float) if len(err_aligned_deg_log) >= N_core else np.full(N_core, np.nan)

    # Raw instantaneous error (at k)
    err_raw_deg_arr = ref_now_deg_arr - meas_deg_arr

    meas_lag1_deg_arr = np.full_like(meas_deg_arr, np.nan)
    if N_core >= 2:
        meas_lag1_deg_arr[:-1] = meas_deg_arr[1:]

    # Ragged horizon logs: slice with N_core too
    ref_seq_deg_obj    = np.array(ref_seq_deg_log[:N_core], dtype=object)
    theta_pred_deg_obj = np.array(theta_pred_deg_log[:N_core], dtype=object)
    u_seq_deg_s_obj    = np.array(u_seq_deg_s_log[:N_core], dtype=object)
    psi_pred_rad_obj   = np.array(psi_pred_rad_log[:N_core], dtype=object)

    # RMSE on aligned error
    fin = np.isfinite(err_aligned_deg_arr)
    rmse_deg = float(np.sqrt(np.mean(err_aligned_deg_arr[fin]**2))) if fin.any() else float("nan")
    print(f"[METRIC] RMSE over run (aligned): {rmse_deg:.3f} deg")

    # --- saving: iterate over N_core, not N ---
    with open(RUN_DIR / "scalars.csv", "w") as f:
        f.write("t,ref_now_deg,ref_aligned_deg,meas_deg,meas_lag1_deg,err_raw_deg,err_aligned_deg,u0_deg_s,psi_now_rad,psi_cmd_rad,qp_infeasible,rmse_aligned\n")
        for i in range(N_core):
            f.write(
                f"{t_arr[i]:.6f},{ref_now_deg_arr[i]:.6f},{ref_aligned_deg_arr[i]:.6f},"
                f"{meas_deg_arr[i]:.6f},{meas_lag1_deg_arr[i]:.6f},"
                f"{err_raw_deg_arr[i]:.6f},{err_aligned_deg_arr[i]:.6f},"
                f"{u0_deg_s_arr[i]:.6f},{psi_now_rad_arr[i]:.6f},{psi_cmd_rad_arr[i]:.6f},"
                f"{qp_infeas_arr[i]},{rmse_deg:.6f}\n"
            )

    # NPZ also uses N_core
    np.savez(
        RUN_DIR / "mpc_rollout_logs.npz",
        t=t_arr,
        ref_now_deg=ref_now_deg_arr,
        ref_aligned_deg=ref_aligned_deg_arr,
        meas_deg=meas_deg_arr,
        meas_lag1_deg=meas_lag1_deg_arr,
        err_raw_deg=err_raw_deg_arr,
        err_aligned_deg=err_aligned_deg_arr,
        u0_deg_s=u0_deg_s_arr,
        psi_now_rad=psi_now_rad_arr,
        psi_cmd_rad=psi_cmd_rad_arr,
        qp_infeasible=qp_infeas_arr,
        ref_seq_deg=ref_seq_deg_obj,
        theta_pred_deg=theta_pred_deg_obj,
        u_seq_deg_s=u_seq_deg_s_obj,
        psi_pred_rad=psi_pred_rad_obj,
        rmse_aligned=rmse_deg,
    )


    # Plots — guard in case N==1 (valid shapes but mostly NaNs are okay)
    # A) Reference vs Measured (measurement lag-1)
    plt.figure(figsize=(10, 6))
    plt.plot(t_arr, ref_now_deg_arr,   label="Reference (deg)", linewidth=2)
    plt.plot(t_arr, meas_lag1_deg_arr, label="Measured (deg, lag 1)", linewidth=1.5)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlabel("Time (s)"); plt.ylabel("Angle (deg)")
    plt.title("Reference vs Measured (measurement one step behind)")
    plt.legend(); plt.tight_layout()
    plt.savefig(RUN_DIR / "plots" / "angles_ref_meas_lag1.png", dpi=200); plt.close()

    # B) Aligned error
    plt.figure(figsize=(10, 4))
    plt.plot(t_arr, err_aligned_deg_arr, label="Aligned Error (deg)")
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlabel("Time (s)"); plt.ylabel("Error (deg)")
    plt.title(f"Aligned Tracking Error (RMSE = {rmse_deg:.2f}°)")
    plt.legend(); plt.tight_layout()
    plt.savefig(RUN_DIR / "plots" / "error_aligned_rmse.png", dpi=200); plt.close()

    # C) ψ plots
    plt.figure(figsize=(10, 4))
    plt.plot(t_arr, np.rad2deg(psi_now_rad_arr), label="ψ_now (deg)")
    plt.plot(t_arr, np.rad2deg(psi_cmd_rad_arr), label="ψ_cmd (deg)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlabel("Time (s)"); plt.ylabel("ψ (deg)")
    plt.title("Joint 6 Command"); plt.legend(); plt.tight_layout()
    plt.savefig(RUN_DIR / "plots" / "j6.png", dpi=200); plt.close()

    # D) Predicted θ rollouts (safe even with small N)
    plt.figure(figsize=(10, 6))
    plt.plot(t_arr, meas_deg_arr,    color="k", linewidth=1.0, label="θ_meas (deg)")
    plt.plot(t_arr, ref_now_deg_arr, color="C0", linewidth=1.2, label="θ_ref (deg)")
    stride = max(1, len(t_arr)//20) if len(t_arr) else 1
    for i in range(0, len(t_arr), stride):
        if i < len(theta_pred_deg_log):
            tp = theta_pred_deg_log[i]
            th = t_arr[i] + (np.arange(1, len(tp)+1) * float(controller.dt))
            plt.plot(th, tp, alpha=0.30)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlabel("Time (s)"); plt.ylabel("Angle (deg)")
    plt.title("Predicted θ Rollouts (overlaid)")
    plt.legend(); plt.tight_layout()
    plt.savefig(RUN_DIR / "plots" / "theta_rollouts.png", dpi=200); plt.close()

    # E) RMSE bar
    plt.figure(figsize=(4, 4))
    plt.bar(["RMSE (deg, aligned)"], [rmse_deg])
    plt.tight_layout()
    plt.savefig(RUN_DIR / "plots" / "rmse_bar_aligned.png", dpi=200); plt.close()

    print(f"[LOG] Plots saved under {RUN_DIR/'plots'}")




if __name__ == "__main__":
    main()
