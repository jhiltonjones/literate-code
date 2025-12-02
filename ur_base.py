import sys
import time
import logging
from spring_mass_system import mag_pose_rotated
from transformations import get_point
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config
import numpy as np

def list_to_setp(setp, lst, offset=0):
    """ Converts a list into RTDE input registers, allowing different sets with offsets. """
    for i in range(6):
        setp.__dict__[f"input_double_register_{i + offset}"] = lst[i]
    return setp

ROBOT_HOST = "192.168.56.101"
ROBOT_PORT = 30004
CONFIG_XML = "control_loop_configuration.xml"
FREQUENCY = 125 

JOINT_TARGET = [-0.4221299330340784, -2.071312566796774, -1.3693373203277588, -1.282651738529541, 1.5822639465332031, -0.4940570036517542]
TCP_TARGET =  [0.7985173296917242, -0.538880495640068, 0.4502643054124873, -1.9773907947518439, 2.4202883191517617, -0.016116851148178644]

def main():
    logging.getLogger().setLevel(logging.INFO)


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
        JOINT_TARGET2 = state.actual_q
        if state.output_bit_registers0_to_31:  
            print('Boolean 1 is True, proceeding to mode 1\n')
            break

    print("-------Executing moveJ -----------\n")

    watchdog.input_int_register_0 = 1
    con.send(watchdog)
    state = con.receive()
    JOINT_TARGET2 = state.actual_q
    print(f"Joints are {JOINT_TARGET2}")
    list_to_setp(setp, JOINT_TARGET, offset=6)
    con.send(watchdog)
    print(f"Target joints are: {state.target_q}")
    con.send(setp)
    time.sleep(0.5)

    while True:
        # print(f'Waiting for moveJ()1 to finish start...')
        state = con.receive()
        con.send(watchdog)
        if not state.output_bit_registers0_to_31:
            print('Start completed\n')
            break

    # # 1) write TCP target FIRST
    # list_to_setp(setp, TCP_TARGET, offset=0)
    # con.send(watchdog)

    # 2) THEN tell URScript to enter mode 2
    watchdog.input_int_register_0 = 2
    con.send(watchdog)
    pose_target = get_point(-10,60)
    list_to_setp(setp, pose_target, offset=0)
    con.send(setp)

    # 3) now wait for it to clear the boolean when done
    while True:
        state = con.receive()
        con.send(watchdog)
        if not state.output_bit_registers0_to_31:
            print('Next completed, proceeding to feedback check\n')
            break



    watchdog.input_int_register_0 = 3
    con.send(watchdog)
    state = con.receive()
    print("Joint State:", state.actual_q)
    print("Mode 3 sent — robot should move to Halt section now.")



if __name__ == "__main__":
    main()
