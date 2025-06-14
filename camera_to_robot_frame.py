import numpy as np
from tip_w_spline import below_or_above
from image_capture import capture_image
from new_cam import new_capture
from new_tip_finder import detect_rod_tip_darkest_right
from june_tip_finder import detect_rod_tip_yellow_right
def pixel_to_robot_frame(pixel_x, pixel_y):
    # Reference: known pixel and robot coordinates
    ref_pixel = np.array([181.84, 253.27])             # pixel_x, pixel_y
    ref_robot = np.array([0.9133455802627718, -0.25720637440350974])          # robot_x, robot_y

    # Scale conversion
    scale_pixels_per_mm = 2.00
    mm_per_pixel = 1 / scale_pixels_per_mm

    # Compute pixel deltas
    dx_px = pixel_x - ref_pixel[0]  # pixel_x ↔ robot_y (inverted)
    dy_px = pixel_y - ref_pixel[1]  # pixel_y ↔ robot_x (inverted)

    # Convert to mm, then to meters
    dx_m = dx_px * mm_per_pixel / 1000.0   # invert for robot_y
    dy_m = dy_px * mm_per_pixel / 1000.0   # invert for robot_x

    # Apply offset
    robot_x = ref_robot[0] + dy_m   # pixel_y → robot_x
    robot_y = ref_robot[1] + dx_m   # pixel_x → robot_y

    return robot_x, robot_y

if __name__ == "__main__":
    # image = capture_image()
    # tip, rod_pos, signed_distance_mm, desired_point = below_or_above(image, False)
    image_path = new_capture()
    # tip, rod_pos, error, desired_point = detect_rod_tip_darkest_right(image_path, graph=False)
    tip, rod_pos, error, desired_point, _,_ = detect_rod_tip_yellow_right(image_path, graph=False)


    print(f"Rod pixel position: {rod_pos}")
    
    robotposx, robotposy = pixel_to_robot_frame(rod_pos[0], rod_pos[1])
    print(robotposy)
    print(f"Robot frame position: x = {robotposx:.5f}, y = {robotposy:.5f}")
