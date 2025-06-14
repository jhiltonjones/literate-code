import numpy as np
from tip_w_spline import below_or_above
from image_capture import capture_image
from camera_to_robot_frame import pixel_to_robot_frame
from tip_w_spline import below_or_above
from image_capture import capture_image
from inverse_pos import inverse_pos_calc
from tip_angle_predictive import below_or_above2
from new_cam import new_capture
from new_finder2 import detect_rod_tip_darkest_right
from june_tip_finder import detect_rod_tip_yellow_right
from june_inverse_jacobians import sympy_solver
# def joint_rad_to_custom_degrees(theta_rad):
#     rad_0_deg = -0.154093090687887
#     rad_90_deg = 1.5742939710617065

#     # Solve for linear mapping: degrees = m * theta + b
#     m = (90 - 0) / (rad_90_deg - rad_0_deg)
#     b = -m * rad_0_deg  # Since 0 = m * rad_0_deg + b

#     custom_degrees = m * theta_rad + b
#     return custom_degrees
# def custom_degrees_to_joint_rad(custom_deg):
#     rad_0_deg = -0.154093090687887
#     rad_90_deg = 1.5742939710617065

#     # Linear mapping parameters
#     m = (90 - 0) / (rad_90_deg - rad_0_deg)
#     b = -m * rad_0_deg

#     # Invert the linear mapping
#     theta_rad = (custom_deg - b) / m
#     return theta_rad

import numpy as np

def joint_rad_to_custom_degrees(theta_rad, x_alignment_deg):
    """
    Convert joint angle in radians to custom degrees based on rod's x-alignment in degrees.
    Assumes rad_0_deg = 4.153112888336182 when aligned with x-axis.
    """
    rad_0_deg = 4.153112888336182
    rad_90_deg = rad_0_deg + np.pi / 2  # Approximate 90° in radians = π/2

    m = (90) / (rad_90_deg - rad_0_deg)
    b = x_alignment_deg - m * rad_0_deg  # shift so that rad_0_deg → x_alignment_deg

    return m * theta_rad + b


def custom_degrees_to_joint_rad(custom_deg, x_alignment_deg):
    """
    Convert custom degrees back to joint radians based on rod's x-alignment.
    """
    rad_0_deg = 4.153112888336182
    rad_90_deg = rad_0_deg + np.pi / 2

    m = (90) / (rad_90_deg - rad_0_deg)
    b = x_alignment_deg - m * rad_0_deg

    return (custom_deg - b) / m


def robot_to_inverse_frame(robot_x, robot_y):
    """
    Converts a robot-frame position to the inverse kinematics frame.
    This should be the position of the rod in the robot frame. The inverse frame should give 0,0 in the original position.
    The inverse frame is there for the jacobian to be computed.
    The jacobian will then return the positon of the magnet to produce the bending required.

    Returns:
        inv_x, inv_y (in meters)
    """
    # origin_robot_x = 0.30998441861443216
    # origin_robot_y = 0.491167008966276
    origin_robot_x = 0.88621
    origin_robot_y = -0.19763
    inv_x = robot_x - origin_robot_x
    inv_y = robot_y - origin_robot_y
    inv_x = -1* inv_x
    inv_y = -1* inv_y

    return inv_x, inv_y
def inverse_to_robot_frame(inv_x, inv_y):
    """
    Converts a position from inverse kinematics frame back to robot frame.
    The IK frame origin is defined at (-0.013964088189107533, 0.36354044542182157) in robot coordinates.

    Returns:
        robot_x, robot_y
    """
    origin_robot_x = 0.88621
    origin_robot_y = -0.19763

    robot_x = origin_robot_x - inv_x
    robot_y = origin_robot_y - inv_y

    return robot_x, robot_y
def position_mapping(rod_pos, robot_x, robot_y, rotation, des_angle, x_alignment):
    # Step 1: Get catheter tip in robot frame
    # image_path = capture_image()
    # tip, rod_pos, error, desired_point = below_or_above(image_path, False)
    tip_pixel_x, tip_pixel_y = rod_pos
    catheter_robot_x, catheter_robot_y = pixel_to_robot_frame(tip_pixel_x, tip_pixel_y)
    catheter_robot_x_inv, catheter_robot_y_inv = robot_to_inverse_frame(catheter_robot_x, catheter_robot_y)
    robot_pos_x_inv, robot_pos_y_inv = robot_to_inverse_frame(robot_x, robot_y)
    print(f'Rod position in robot frame = {catheter_robot_x, catheter_robot_y}')
    print(f'Rod position in inverse frame = {catheter_robot_x_inv, catheter_robot_y_inv}')
    print(f'Robot position in inverse frame = {robot_pos_x_inv, robot_pos_y_inv}')
    # Step 2: Get current magnet robot frame position
    # robot_x = 0.2740038863899023
    # robot_y = 0.41199797281472317

    inv_x, inv_y = robot_to_inverse_frame(robot_x, robot_y)
    print(f"Inverse Frame: x = {inv_x:.3f}, y = {inv_y:.3f}")
    diff_x = abs(catheter_robot_x_inv - inv_x)
    diff_y = abs(catheter_robot_y_inv - inv_y)
    print(f"Difference x: {diff_x}, y: {diff_y}")
    deg = joint_rad_to_custom_degrees(rotation, x_alignment)

    x_var = np.array([diff_y, diff_x, deg])
    # theta_deg_out, theta_c_desired, x_calc_pos, y_calc_pos, rotation_calc2 = inverse_pos_calc(des_angle, x_var)
    y_calc_pos, x_calc_pos, rotation_calc2 = sympy_solver(des_angle, x_var, des_angle)
    x_calc_pos = 1* x_calc_pos
    # if np.rad2deg(des_angle) > 15:
    #     print(f"yposition = {y_calc_pos}")
    #     y_calc_pos += 0.15
    #     # This is to align the field with the vessel branch 
    #     print(f"yposition adjusted = {y_calc_pos}")
    # elif np.rad2deg(des_angle) < -15:
    #     y_calc_pos -=0.1


    print(f'Degrees: {rotation_calc2} Radians: {des_angle}')
    final_rotation = custom_degrees_to_joint_rad(rotation_calc2, x_alignment)
    finalp_in_catheter_x = catheter_robot_x_inv + x_calc_pos
    finalp_in_catheter_y = catheter_robot_y_inv + y_calc_pos 
    print(f"Desired position in inverse kinematics relative to catehter: x = {finalp_in_catheter_x:.3f}, y = {finalp_in_catheter_y:.3f}, Rotation = {rotation_calc2}")
    mag_pos_final_x, mag_pos_final_y = inverse_to_robot_frame(finalp_in_catheter_x, finalp_in_catheter_y)
    # mag_pos_final_y = -1*mag_pos_final_y
    # mag_pos_final_x = -1*mag_pos_final_x
    print(f"Robot Frame Reconstructed: x = {mag_pos_final_x:.3f}, y = {mag_pos_final_y:.3f}")

    return mag_pos_final_x, mag_pos_final_y, final_rotation


if __name__ == "__main__":
    # image_path = capture_image()
    image_path = new_capture()
    # tip, rod_pos, error, desired_point, alignement = detect_rod_tip_darkest_right(image_path, False)
    tip, rod_pos, error, desired_point, alignement, x_alignment = detect_rod_tip_yellow_right(image_path, False)

    x, y, deg_out = position_mapping(rod_pos, 0.8899670582219864, -0.4023449026980444, -0.16631919542421514, np.deg2rad(alignement), x_alignment)
    print(f"Robot Frame Reconstructed: x = {x:.3f}, y = {y:.3f}, Rotation = {deg_out}")
    print(np.rad2deg(deg_out))

