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
    -0.45088225999941045, -1.9217144451537074, -1.6537089347839355,
    -1.148652271633484,   1.638681983947754,    0.9267773628234863
]
JOINT_TARGET3 = [
    -0.45088225999941045, -1.9217144451537074, -1.6537089347839355,
    -1.248652271633484,   1.738681983947754,    0.9267773628234863
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
    # --- MODE 2: multi-waypoint min-jerk path ---
    watchdog.input_int_register_0 = 2
    con.send(watchdog)

    DT = 1.0 / FREQUENCY
    deg = lambda d: d * 3.141592653589793 / 180.0

    q0 = JOINT_TARGET
    # Define via-points as offsets around q0 (keep safe!)
    SCALE = 0.3  # 30% of the original amplitudes

    wps = [
        q0,
        [q0[0] + deg(12*SCALE), q0[1] - deg(6*SCALE),  q0[2],              q0[3],              q0[4],              q0[5]],
        [q0[0] - deg(8*SCALE),  q0[1],                 q0[2] + deg(10*SCALE), q0[3] - deg(5*SCALE), q0[4],           q0[5]],
        [q0[0],                 q0[1] + deg(12*SCALE), q0[2],              q0[3] + deg(8*SCALE),  q0[4] - deg(6*SCALE), q0[5]],
        q0,
    ]


    def min_jerk(qA, qB, T, hz):
        import math
        N = max(2, int(T * hz))
        for k in range(N):
            tau = k / (N - 1)
            s = 10*tau**3 - 15*tau**4 + 6*tau**5
            yield [a + s*(b - a) for a, b in zip(qA, qB)]

    def stream_sequence(seq, dt):
        import time
        next_tick = time.perf_counter()
        for q in seq:
            for i in range(6):
                setattr(setp, f"input_double_register_{12 + i}", float(q[i]))
            if "input_bit_registers0_to_31" in setp_names:
                setp.input_bit_registers0_to_31 = 0
            con.send(setp)
            next_tick += dt
            rem = next_tick - time.perf_counter()
            if rem > 0: time.sleep(rem)

    # travel each segment smoothly (3 s per segment; tweak as you like)
    for a, b in zip(wps[:-1], wps[1:]):
        stream_sequence(min_jerk(a, b, T=3.0, hz=FREQUENCY), DT)

    # exit Mode 2
    watchdog.input_int_register_0 = 3
    con.send(watchdog)



if __name__ == "__main__":
    main()
