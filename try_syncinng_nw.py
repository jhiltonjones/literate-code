import sys
import logging
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config
import time
import numpy as np
import matplotlib.pyplot as plt
from trajectory_planner_line import PathPlanTranslation 
from rotation_matrix import T, transform_point
from bfgs_minimise import alpha_star, alpha_star_deg, compute_angle, length_c, x_p, y_p, p_norm1
from constants import d, h, theta_l, EI, x_basis, y_basis
from PID_control import PIDController
def setp_to_list(setp, offset=0):
    return [setp.__dict__[f"input_double_register_{i + offset}"] for i in range(6)]


def list_to_setp(setp, lst, offset=0):
    """ Converts a list into RTDE input registers, allowing different sets with offsets. """
    for i in range(6):
        setp.__dict__[f"input_double_register_{i + offset}"] = lst[i]
    return setp

# ------------- Robot Communication Setup -----------------
ROBOT_HOST = '192.168.56.101'
ROBOT_PORT = 30004
config_filename = 'control_loop_configuration.xml'  

logging.getLogger().setLevel(logging.INFO)

conf = rtde_config.ConfigFile(config_filename)
state_names, state_types = conf.get_recipe('state')  
setp_names, setp_types = conf.get_recipe('setp')  
watchdog_names, watchdog_types = conf.get_recipe('watchdog')


con = rtde.RTDE(ROBOT_HOST, ROBOT_PORT)
connection_state = con.connect()

while connection_state != 0:
    time.sleep(0.5)
    connection_state = con.connect()
print("---------------Successfully connected to the robot-------------\n")

con.get_controller_version()


FREQUENCY = 500  
con.send_output_setup(state_names, state_types, FREQUENCY)
setp = con.send_input_setup(setp_names, setp_types)
watchdog = con.send_input_setup(watchdog_names, watchdog_types)

for i in range(12): 
    setattr(setp, f"input_double_register_{i}", 0)


setp.input_bit_registers0_to_31 = 0
watchdog.input_int_register_0 = 0


if not con.send_start():
    sys.exit()


transformed_points = transform_point(T, d, h)
x_robotic_arm = transformed_points[0]
y_robotic_arm = transformed_points[1]

start_point = [-0.31563740358348213, 0.451164987228323, 0.38704760524680776, -2.195089565641003, 2.2236842260238285, -0.012538948420594043]

waypoints = [
    [-0.3156422082847773, 0.4511485435638499, 0.1869893060870562, -0.881634584042421, 2.9958190256705133, 0.0013891383636546828],
    [x_robotic_arm, y_robotic_arm, 0.1870643751382633, -3.1129835149573597, -0.39044182826995194, -0.02877771799883438],
    # [0.11561346376380156, 0.4392848297605487, 0.3870643751382633, -3.1129835149573597, -0.39044182826995194, -0.02877771799883438], # Start
    # [0.01561346376380156, 0.8392848297605487, 0.3870643751382633, -3.1129835149573597, -0.39044182826995194, -0.02877771799883438],
    # [-0.31561346376380156, 0.8392848297605487, 0.3870643751382633, -3.1129835149573597, -0.39044182826995194, -0.02877771799883438],
    # [-0.31561346376380156, 0.4392848297605487, 0.3870643751382633, -3.1129835149573597, -0.39044182826995194, -0.02877771799883438],
    # [0.11561346376380156, 0.4392848297605487, 0.3870643751382633, -3.1129835149573597, -0.39044182826995194, -0.02877771799883438], # Start
    # [0.01561346376380156, 0.8392848297605487, 0.3870643751382633, -3.1129835149573597, -0.39044182826995194, -0.02877771799883438],
    # [-0.31561346376380156, 0.8392848297605487, 0.3870643751382633, -3.1129835149573597, -0.39044182826995194, -0.02877771799883438],
    # [-0.31561346376380156, 0.4392848297605487, 0.3870643751382633, -3.1129835149573597, -0.39044182826995194, -0.02877771799883438]
    # [-0.31561346376380156, 0.8392848297605487, 0.3870643751382633, -3.1129835149573597, -0.39044182826995194, -0.02877771799883438],
    # [0.01561346376380156, 0.8392848297605487, 0.3870643751382633, -3.1129835149573597, -0.39044182826995194, -0.02877771799883438] 
]

orientation_const = waypoints[0][3:]
trajectory_time_total = 3
trajectory_time_per_segment = trajectory_time_total / (len(waypoints) - 1) 
state = con.receive()
tcp1 = state.actual_TCP_pose
print(tcp1)
start_point_list = setp_to_list(setp, offset=6)
waypoints_list = setp_to_list(setp, offset=0) 
position_list = setp_to_list(setp, offset=6)  


while True:
    print('Boolean 1 is False, please click CONTINUE on the Polyscope')
    state = con.receive()
    con.send(watchdog)
    if state.output_bit_registers0_to_31:  
        print('Boolean 1 is True, proceeding to mode 1\n')
        break
print("-------Executing moveJ start -----------\n")


watchdog.input_int_register_0 = 1
con.send(watchdog)
list_to_setp(setp, waypoints[0], offset=0) 
con.send(setp) 

while True:
    print('Waiting for moveJ() first to finish')
    state = con.receive()
    con.send(watchdog)
    if not state.output_bit_registers0_to_31:
        print('Proceeding to mode 2\n')
        break
print("-------Executing moveJ -----------\n")

watchdog.input_int_register_0 = 1
con.send(watchdog)
list_to_setp(setp, waypoints[0], offset=0) 
con.send(setp) 

while True:
    print('Waiting for moveJ() first to finish')
    state = con.receive()
    con.send(watchdog)
    if not state.output_bit_registers0_to_31:
        print('Proceeding to mode 2\n')
        break


print("-------Executing servoJ  -----------\n")
watchdog.input_int_register_0 = 2
con.send(watchdog)
trajectory_time = 8  
dt = 1/500  # 500 Hz    # frequency
plotter = True
if plotter:
    time_plot = []

    min_jerk_x = []
    min_jerk_y = []
    min_jerk_z = []

    min_jerk_vx = []
    min_jerk_vy = []
    min_jerk_vz = []

    px = []
    py = []
    pz = []

    vx = []
    vy = []
    vz = []

 
current_waypoint_index = 0
t_start = time.time()

while current_waypoint_index < len(waypoints) - 1:
    t_segment_start = time.time()
    start_point = waypoints[current_waypoint_index]
    end_point = waypoints[current_waypoint_index + 1]

    planner = PathPlanTranslation(start_point, end_point, trajectory_time_per_segment)  

    while time.time() - t_segment_start < trajectory_time_per_segment:
        t_current = time.time() - t_segment_start
        state = con.receive()

        if state.runtime_state > 1:
            [position_ref, lin_vel_ref, acceleration_ref] = planner.trajectory_planning(t_current)
            current_pose = state.actual_TCP_pose
            current_speed = state.actual_TCP_speed
            pose = position_ref.tolist() + orientation_const
            list_to_setp(setp, pose)
            con.send(setp)



          
            if plotter:
                time_plot.append(time.time() - t_start)

                min_jerk_x.append(position_ref[0])
                min_jerk_y.append(position_ref[1])
                min_jerk_z.append(position_ref[2])

                min_jerk_vx.append(lin_vel_ref[0])
                min_jerk_vy.append(lin_vel_ref[1])
                min_jerk_vz.append(lin_vel_ref[2])

                px.append(current_pose[0])
                py.append(current_pose[1])
                pz.append(current_pose[2])

                vx.append(current_speed[0])
                vy.append(current_speed[1])
                vz.append(current_speed[2])

    current_waypoint_index += 1  

print(f"It took {time.time()-t_start}s to execute the servoJ")

state = con.receive()
time.sleep(0.5)  



pid = PIDController(Kp=0.5, Ki=0.1, Kd=0.05, dt=0.1)


max_attempts = 10
rotation_step = 0.05 
vessel_branch_target_angle = 45 

print("-------Executing moveJ with PID -----------\n")

watchdog.input_int_register_0 = 3 

for attempt in range(max_attempts):
    print(f"Iteration {attempt + 1}")
    

    state = con.receive()
    actual_position = state.actual_q
    
    if attempt == 0:
        print(f"Initial move: Applying rotation angle of {alpha_star}")
        position = [float(joint) for joint in actual_position] 
        position[5] += alpha_star 
        print(f"Moving magnet to position: {position}")
        list_to_setp(setp, position, offset=6)
        con.send(setp)
        time.sleep(0.5) 
        con.send(watchdog)

    else:
        print(f"Initial move: Applying rotation angle of {rotation_step}")

        position = [float(joint) for joint in actual_position] 
        temp = alpha_star
        new_alpha_star = compute_angle(EI, theta_l, length_c, x_p, y_p, x_basis, y_basis, p_norm1)

        position[5] += rotation_step 

        print(f"Moving magnet to position: {position}")
        list_to_setp(setp, position, offset=6)
        con.send(setp)
        time.sleep(0.5) 
        con.send(watchdog)


    i = 0
    while True:
        print(f'Waiting for moveJ() to finish... {i}')
        state = con.receive()
        con.send(watchdog)
        i += 1
        if not state.output_bit_registers0_to_31:
            print('MoveJ completed, proceeding to feedback check\n')
            break
    

    state = con.receive()  
    print("Checking feedback...")

    catheter_tip_position = np.random.uniform(44.8, 45.2) 
    position_error = catheter_tip_position - vessel_branch_target_angle
    print(f"Catheter Tip Position: {catheter_tip_position}, Position Error: {position_error}")

    
    if abs(position_error) >= 0.005: 
        rotation_adjustment = pid.update(position_error)
        print(f"PID Rotation Adjustment: {rotation_adjustment}")

 
        state = con.receive()
        actual_position = state.actual_q
        position = [float(joint) for joint in actual_position]  
        position[5] += rotation_adjustment 

        print(f"Adjusting magnet based on PID: {position}")
        list_to_setp(setp, position, offset=6)
        con.send(setp)
        time.sleep(0.5)  
        con.send(watchdog)

        while True:
            print('Waiting for PID-adjusted moveJ() to finish...')
            state = con.receive()
            con.send(watchdog)
            if not state.output_bit_registers0_to_31:
                print('PID-adjusted MoveJ completed.\n')
                break
    else:
        print("Error is minimal. No further adjustments needed.")
        break 

print("Final movement completed.")
print('--------------------')
print("Actual joint position after moveJ:", state.actual_q)
print("Actual TCP pose after moveJ:", state.actual_TCP_pose)
print("Moved to position: ", d,h)
print("With a rotation of: ", alpha_star_deg)
print("With a steering angle of: ", np.rad2deg(theta_l))

watchdog.input_int_register_0 = 4
con.send(watchdog)

con.send_pause()
con.disconnect()


if plotter:
    # ----------- Position Plots -------------
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    axs[0].plot(time_plot, min_jerk_x, label="x_min_jerk")
    axs[0].plot(time_plot, px, label="x_robot")
    axs[0].set_ylabel('Position in x [m]')
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(time_plot, min_jerk_y, label="y_min_jerk")
    axs[1].plot(time_plot, py, label="y_robot")
    axs[1].set_ylabel('Position in y [m]')
    axs[1].legend()
    axs[1].grid()

    axs[2].plot(time_plot, min_jerk_z, label="z_min_jerk")
    axs[2].plot(time_plot, pz, label="z_robot")
    axs[2].set_ylabel('Position in z [m]')
    axs[2].set_xlabel('Time [sec]')
    axs[2].legend()
    axs[2].grid()

    plt.show()

    # ----------- Velocity Plots -------------
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    axs[0].plot(time_plot, min_jerk_vx, label="vx_min_jerk")
    axs[0].plot(time_plot, vx, label="vx_robot")
    axs[0].set_ylabel('Velocity in x [m/s]')
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(time_plot, min_jerk_vy, label="vy_min_jerk")
    axs[1].plot(time_plot, vy, label="vy_robot")
    axs[1].set_ylabel('Velocity in y [m/s]')
    axs[1].legend()
    axs[1].grid()

    axs[2].plot(time_plot, min_jerk_vz, label="vz_min_jerk")
    axs[2].plot(time_plot, vz, label="vz_robot")
    axs[2].set_ylabel('Velocity in z [m/s]')
    axs[2].set_xlabel('Time [sec]')
    axs[2].legend()
    axs[2].grid()

    plt.show()