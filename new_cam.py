# import cv2

# def new_capture(filename='focused_image.jpg', focus=255):
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         raise RuntimeError("Cannot open camera")

#     cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
#     cap.set(cv2.CAP_PROP_FOCUS, focus)

#     ret, frame = cap.read()
#     cap.release()

#     if ret:
#         cv2.imwrite(filename, frame)
#         print(f"Focused image captured and saved as {filename}")
#         return filename
#     else:
#         raise RuntimeError("Failed to capture image")
# if __name__ == "__main__":
#     new_capture()

import cv2
import matplotlib.pyplot as plt
import numpy as np

def new_capture(filename='focused_image.jpg', focus=255):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    # Try disabling autofocus (not all cameras support this)
    if cap.set(cv2.CAP_PROP_AUTOFOCUS, 0):
        print("Autofocus disabled.")
    else:
        print("Warning: Could not disable autofocus (unsupported).")

    # Try setting manual focus (value typically from 0 to 255)
    if cap.set(cv2.CAP_PROP_FOCUS, float(focus)):
        print(f"Focus set to {focus}.")
    else:
        print("Warning: Could not set focus manually (unsupported).")

    # Optional: Disable auto exposure and auto white balance for consistent lighting
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual mode on many webcams
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)         # May need tuning per device

    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)

    # Wait briefly to let camera apply new settings
    for _ in range(5):
        cap.read()

    # Capture final frame
    ret, frame = cap.read()
    cap.release()

    if ret and frame is not None:
        cv2.imwrite(filename, frame)
        print(f"Focused image captured and saved as {filename}")
        return filename
    else:
        raise RuntimeError("Failed to capture image")
# ─── IMAGE PROCESSING ────────────────────────────────────────────────────
def compute_signed_angle(v1, v2):
    """Returns the signed angle in degrees from v1 to v2 (positive = CCW, negative = CW)"""
    angle1 = np.arctan2(v1[1], v1[0])
    angle2 = np.arctan2(v2[1], v2[0])
    angle_rad = angle2 - angle1
    angle_deg = np.degrees(angle_rad)

    # Normalize to [-180, 180]
    if angle_deg > 180:
        angle_deg -= 360
    elif angle_deg < -180:
        angle_deg += 360

    return angle_deg

def detect_red_points_and_angle(image_path, show=True):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    red_ranges = [
        (np.array([0, 50, 50]), np.array([10, 255, 255])),
        (np.array([160, 50, 50]), np.array([180, 255, 255]))
    ]

    red_mask = None
    for lower_red, upper_red in red_ranges:
        temp_mask = cv2.inRange(image_hsv, lower_red, upper_red)
        red_mask = temp_mask if red_mask is None else cv2.bitwise_or(red_mask, temp_mask)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 2:
        raise ValueError("Less than two red points detected!")

    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    red_centers = []
    for cnt in sorted_contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            red_centers.append((cx, cy))
            cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)

    pt1, pt2 = red_centers
    vector = np.array(pt2) - np.array(pt1)
    reference = np.array([1, 0])  # x-axis

    angle = compute_signed_angle(reference, vector)

    if show:
        cv2.line(image, pt1, pt2, (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f"Beam Angle: {angle:.2f}°")
        plt.axis("off")
        plt.show()

    return pt1, pt2, angle
if __name__ == "__main__":
    new_capture()
    image_path = "/home/jack/literate-code/focused_image.jpg"
    pt1, pt2, angle = detect_red_points_and_angle(image_path)
