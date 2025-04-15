import numpy as np
from tip_w_spline import below_or_above
from image_capture import capture_image

def pixel_to_robot_frame(pixel_x, pixel_y):
    # Reference: known pixel and robot coordinates
    ref_pixel = np.array([1094.98, 456.50])             # pixel_x, pixel_y
    ref_robot = np.array([-0.013964088189107533, 0.36354044542182157])          # robot_x, robot_y

    # Scale conversion
    scale_pixels_per_mm = 9.55
    mm_per_pixel = 1 / scale_pixels_per_mm

    # Compute pixel deltas
    dx_px = pixel_x - ref_pixel[0]  # pixel_x ↔ robot_y (inverted)
    dy_px = pixel_y - ref_pixel[1]  # pixel_y ↔ robot_x (inverted)

    # Convert to mm, then to meters
    dx_m = -dx_px * mm_per_pixel / 1000.0   # invert for robot_y
    dy_m = dy_px * mm_per_pixel / 1000.0   # invert for robot_x

    # Apply offset
    robot_x = ref_robot[0] + dy_m   # pixel_y → robot_x
    robot_y = ref_robot[1] + dx_m   # pixel_x → robot_y

    return robot_x, robot_y

if __name__ == "__main__":
    image = capture_image()
    tip, rod_pos, signed_distance_mm, desired_point = below_or_above(image, False)
    print(f"Rod pixel position: {rod_pos}")
    
    robotposx, robotposy = pixel_to_robot_frame(rod_pos[0], rod_pos[1])
    print(f"Robot frame position: x = {robotposx:.5f}, y = {robotposy:.5f}")
