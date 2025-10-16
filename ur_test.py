import sys
import time
import logging

import rtde.rtde as rtde
import rtde.rtde_config as rtde_config
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
    -0.4110644499408167,
    -1.8883592091002406,
    -1.699528455734253,
    -1.1374167960933228,
    1.5391621589660645,
    -3.4619577566729944,
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

    joint_pos = JOINT_TARGET.copy()
    increment = 0  # radians (~3 degrees)

    for i in range(4):
        
        joint_pos[2] += increment  # increase joint 6
        list_to_setp(setp, joint_pos, offset=6)
        con.send(setp)
        print(f"Increment {i+1}: joint6 = {joint_pos[5]:.4f} rad")

        # give the robot time to move
        time.sleep(1.0)

        # optional wait for busy bit to clear before next increment
        while True:
            state = con.receive()
            con.send(watchdog)
            if not state.output_bit_registers0_to_31:
                break
            time.sleep(0.005)

    print("All three joint6 increments completed.\n")

    # proceed with next mode
    watchdog.input_int_register_0 = 3
    con.send(watchdog)
    print("Mode 3 sent — robot should move to Halt section now.")
    con.send_pause()
    con.disconnect()


if __name__ == "__main__":
    main()
