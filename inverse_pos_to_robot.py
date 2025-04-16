import numpy as np
from tip_w_spline import below_or_above
from image_capture import capture_image
from camera_to_robot_frame import pixel_to_robot_frame
from tip_w_spline import below_or_above
from image_capture import capture_image
from inverse_pos import inverse_pos_calc
from tip_angle_predictive import below_or_above2

def joint_rad_to_custom_degrees(theta_rad):
    # Known calibration points
    rad_0_deg = -2.536228958760397
    rad_90_deg = -1.1406424681292933

    # Solve for linear mapping: degrees = m * theta + b
    m = (90 - 0) / (rad_90_deg - rad_0_deg)
    b = -m * rad_0_deg  # Since 0 = m * rad_0_deg + b

    custom_degrees = m * theta_rad + b
    return custom_degrees
def custom_degrees_to_joint_rad(custom_deg):
    # Known calibration points
    rad_0_deg = -2.536228958760397
    rad_90_deg = -1.1406424681292933

    # Linear mapping parameters
    m = (90 - 0) / (rad_90_deg - rad_0_deg)
    b = -m * rad_0_deg

    # Invert the linear mapping
    theta_rad = (custom_deg - b) / m
    return theta_rad

def robot_to_inverse_frame(robot_x, robot_y):
    """
    Converts a robot-frame position to the inverse kinematics frame.
    The IK frame origin is defined at (-0.013964088189107533, 0.36354044542182157) in robot coordinates.

    Returns:
        inv_x, inv_y (in meters)
    """
    origin_robot_x = -0.013964088189107533
    origin_robot_y = 0.36354044542182157

    inv_x = robot_x - origin_robot_x
    inv_y = robot_y - origin_robot_y

    return inv_x, inv_y
def inverse_to_robot_frame(inv_x, inv_y):
    """
    Converts a position from inverse kinematics frame back to robot frame.
    The IK frame origin is defined at (-0.013964088189107533, 0.36354044542182157) in robot coordinates.

    Returns:
        robot_x, robot_y
    """
    origin_robot_x = -0.013964088189107533
    origin_robot_y = 0.36354044542182157

    robot_x = origin_robot_x + inv_x
    robot_y = origin_robot_y + inv_y

    return robot_x, robot_y
def position_mapping(rod_pos, robot_x, robot_y, rotation, des_angle):
    # Step 1: Get catheter tip in robot frame
    # image_path = capture_image()
    # tip, rod_pos, error, desired_point = below_or_above(image_path, False)
    tip_pixel_x, tip_pixel_y = rod_pos
    catheter_robot_x, catheter_robot_y = pixel_to_robot_frame(tip_pixel_x, tip_pixel_y)
    catheter_robot_x_inv, catheter_robot_y_inv = robot_to_inverse_frame(catheter_robot_x, catheter_robot_y)
    print(f'Rod position in robot frame = {catheter_robot_x, catheter_robot_y}')
    print(f'Rod position in inverse frame = {catheter_robot_x_inv, catheter_robot_y_inv}')
    # Step 2: Get current magnet robot frame position
    # robot_x = 0.2740038863899023
    # robot_y = 0.41199797281472317

    inv_x, inv_y = robot_to_inverse_frame(robot_x, robot_y)
    print(f"Inverse Frame: x = {inv_x:.3f}, y = {inv_y:.3f}")
    diff_x = abs(catheter_robot_x_inv - inv_x)
    diff_y = abs(catheter_robot_y_inv - inv_y)
    deg = joint_rad_to_custom_degrees(rotation)

    x_var = np.array([diff_x, diff_y, deg])
    # des_angle = np.deg2rad(25)
    theta_deg_out, theta_c_desired, x_calc_pos, y_calc_pos, rotation_calc2 = inverse_pos_calc(des_angle, x_var)
    print(f'Degrees: {rotation_calc2} Radians: {theta_c_desired}')
    final_rotation = custom_degrees_to_joint_rad(rotation_calc2)
    finalp_in_catheter_x = catheter_robot_x_inv + x_calc_pos
    finalp_in_catheter_y = catheter_robot_y_inv + y_calc_pos
    print(f"Desired position in inverse kinematics relative to catehter: x = {finalp_in_catheter_y:.3f}, y = {finalp_in_catheter_x:.3f}, Rotation = {theta_deg_out}")
    mag_pos_final_x, mag_pos_final_y = inverse_to_robot_frame(finalp_in_catheter_y, finalp_in_catheter_x)
    print(f"Robot Frame Reconstructed: x = {mag_pos_final_x:.3f}, y = {mag_pos_final_y:.3f}")

    return mag_pos_final_x, mag_pos_final_y, final_rotation


if __name__ == "__main__":
    image_path = capture_image()
    tip, rod_pos, error, desired_point, alignement = below_or_above2(image_path, False)
    x, y, deg_out = position_mapping(rod_pos, 0.23602152160567344, 0.3550124753628118, 0.32630395889282227, -10)
    print(f"Robot Frame Reconstructed: x = {x:.3f}, y = {y:.3f}, Rotation = {deg_out}")
    print(np.rad2deg(deg_out))

