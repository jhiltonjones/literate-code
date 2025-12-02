import sys
import time
import logging
import math
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config
import numpy as np

ROBOT_HOST = "192.168.56.101"
ROBOT_PORT = 30004
CONFIG_XML = "control_loop_configuration.xml"
FREQUENCY = 25  # Hz

# input_double_registers 6..11 hold the joint target q[0..5]
JOINT_TARGET = [
-0.23611575761903936, -1.4857219022563477, -1.5130348205566406, -1.7188593349852503, -4.753337089215414, -0.4368069807635706]

def write_joint_target(setp, q, offset=6):
    for i in range(6):
        setattr(setp, f"input_double_register_{offset + i}", float(q[i]))

def bit0_is_true(mask: int) -> bool:
    return (mask & 0b1) == 1

def main():
    logging.getLogger().setLevel(logging.INFO)

    # Load RTDE recipes (make sure your XML exposes:
    #  - state: output_bit_registers0_to_31 (and whatever else you need)
    #  - setp:  input_double_register_0..23 (we use 6..11)
    #  - watchdog: input_int_register_0
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

    # Initialize ALL setp fields you'll send
    for i in range(24):
        setattr(setp, f"input_double_register_{i}", 0.0)
    setp.input_bit_registers0_to_31 = 0
    watchdog.input_int_register_0 = 0

    if not con.send_start():
        print("Failed to send_start()")
        sys.exit(1)

    # === Handshake: wait until Polyscope sets Boolean output register 0 True ===
    print("Waiting for operator CONTINUE on pendant (Boolean output reg 0 → True)...")
    while True:
        state = con.receive()
        if state is None:
            continue
        con.send(watchdog)  # keep session alive
        if hasattr(state, "output_bit_registers0_to_31") and bit0_is_true(state.output_bit_registers0_to_31):
            print("Boolean 1 is True → proceeding to mode 1.")
            break

    # === Tell Polyscope to enter Mode 1 (move) and send joint target in float regs 6..11 ===
    print("------- Executing moveJ -----------")
    watchdog.input_int_register_0 = 1
    con.send(watchdog)

    write_joint_target(setp, JOINT_TARGET, offset=6)
    setp.input_bit_registers0_to_31 = 0  # keep it set
    con.send(setp)

    # Give Polyscope a moment to read the registers
    time.sleep(0.2)

    # === Wait until Polyscope clears Boolean output reg 0 back to False after move completes ===
    print("Waiting for moveJ to finish (Boolean output reg 0 → False)...")
    while True:
        state = con.receive()
        if state is None:
            continue
        con.send(watchdog)
        if hasattr(state, "output_bit_registers0_to_31") and not bit0_is_true(state.output_bit_registers0_to_31):
            print("Move complete detected. Proceeding to Mode 3 (halt).")
            break
    # === MODE 2: stream TCP poses (min-jerk), with RTDE keepalive ===
    watchdog.input_int_register_0 = 2
    con.send(watchdog)

    DT = 1.0 / FREQUENCY
    def deg(a): return a * math.pi / 180.0

    # Center pose
    TCP0 = [0.5740579641386485, -0.32218030432668593, 0.7567283131451028, 2.399295395451942, -1.962252123904804, 0.021404631424785873]

    # Amplitudes (keep within workspace; increase slowly once verified)
    AX, AY, AZ = 0.09, 0.07, 0.06       # meters, x/y/z
    ARX, ARY, ARZ = deg(8), deg(7), deg(5)  # radians, rx/ry/rz

    # Incommensurate freqs (complex pattern, smooth motion)
    fx, fy, fz = 0.13, 0.17, 0.11  # Hz
    frx, fry, frz = 0.07, 0.09, 0.05

    # Phase offsets so we don't start with zero velocity on all axes
    phx, phy, phz = 0.0, math.pi/3, math.pi/2
    phrx, phry, phrz = math.pi/4, math.pi/6, math.pi/2

    # Total streaming time (increase for longer runs)
    T_TOTAL = 20.0  # seconds

    t0 = time.perf_counter()
    next_tick = t0
    while (time.perf_counter() - t0) < T_TOTAL:
        t = time.perf_counter() - t0

        # Parametric 3D Lissajous in position
        x = TCP0[0] + AX * math.sin(2*math.pi*fx*t + phx)
        y = TCP0[1] + AY * math.sin(2*math.pi*fy*t + phy)
        z = TCP0[2] + AZ * math.sin(2*math.pi*fz*t + phz)

        # Gentle wrist “weave” in axis–angle (keep small)
        rx = TCP0[3] + ARX * math.sin(2*math.pi*frx*t + phrx)
        ry = TCP0[4] + ARY * math.sin(2*math.pi*fry*t + phry)
        rz = TCP0[5] + ARZ * math.sin(2*math.pi*frz*t + phrz)

        # Send pose to regs 0..5
        pose = [x, y, z, rx, ry, rz]
        for i in range(6):
            setattr(setp, f"input_double_register_{i}", float(pose[i]))
        if "input_bit_registers0_to_31" in setp_names:
            setp.input_bit_registers0_to_31 = 0
        con.send(setp)

        # Keep RTDE session alive + pace exactly
        state = con.receive()
        con.send(watchdog)

        next_tick += DT
        rem = next_tick - time.perf_counter()
        if rem > 0:
            time.sleep(rem)

    # exit Mode 2 (unchanged)
    watchdog.input_int_register_0 = 3
    con.send(watchdog)
if __name__ == "__main__":
    main()
