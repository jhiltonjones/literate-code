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
    -0.45088225999941045, -1.9217144451537074, -1.6537089347839355,
    -1.148652271633484,   1.538681983947754,    0.9267773628234863
]
JOINT_TARGET2 = [
    -0.55088225999941045, -1.9217144451537074, -1.6537089347839355,
    -1.148652271633484,   1.638681983947754,    0.9267773628234863
]
JOINT_TARGET3 = [
    -0.45088225999941045, -1.9217144451537074, -1.6537089347839355,
    -1.248652271633484,   1.838681983947754,    0.9267773628234863
]
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
    # --- MODE 2: increment through joint states using movej (regs 6..11) ---
    watchdog.input_int_register_0 = 2
    con.send(watchdog)

    sequence = [JOINT_TARGET, JOINT_TARGET2, JOINT_TARGET3, JOINT_TARGET]

    PER_MOVE_S = 1.0
    DT = 1.0 / FREQUENCY

    for q in sequence:
        for i in range(6):
            setattr(setp, f"input_double_register_{6 + i}", float(q[i]))  # regs 6..11
        if "input_bit_registers0_to_31" in setp_names:
            setp.input_bit_registers0_to_31 = 0
        con.send(setp)

        t_end = time.perf_counter() + PER_MOVE_S
        while time.perf_counter() < t_end:
            state = con.receive()
            con.send(watchdog)
            time.sleep(DT)

    # exit Mode 2
    watchdog.input_int_register_0 = 3
    con.send(watchdog)



if __name__ == "__main__":
    main()
