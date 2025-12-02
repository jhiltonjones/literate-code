import cv2
import matplotlib.pyplot as plt
import numpy as np

# =====================================================================
# CHECKERBOARD CONFIG  (MUST MATCH WHAT YOU PRINTED!)
# =====================================================================
CHECKERBOARD_SQUARE_SIZE_MM = 5.0  # physical size of one square (mm)
CHECKERBOARD_SQUARES_X = 9         # number of squares horizontally (cols)
CHECKERBOARD_SQUARES_Y = 6         # number of squares vertically (rows)

# OpenCV uses INNER CORNERS, which is one less than squares
CHECKERBOARD_PATTERN_SIZE = (
    CHECKERBOARD_SQUARES_X - 1,    # inner corners along width
    CHECKERBOARD_SQUARES_Y - 1     # inner corners along height
)

# =====================================================================
# CAMERA CAPTURE
# =====================================================================
def new_capture(filename='focused_image.jpg', focus=255):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    # Optional exposure / WB tuning
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual mode on many webcams
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)         # May need tuning per device
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)

    # Let camera settle
    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if ret and frame is not None:
        cv2.imwrite(filename, frame)
        return filename
    else:
        raise RuntimeError("Failed to capture image")

# =====================================================================
# ANGLE UTILS
# =====================================================================
class AngleUnwrapper:
    def __init__(self): 
        self.prev = None

    def __call__(self, a):
        if self.prev is None:
            self.prev = a
            return a
        delta = a - self.prev
        if   delta > 180.0: 
            a -= 360.0
        elif delta < -180.0: 
            a += 360.0
        self.prev = a
        return a

unwrap_angle = AngleUnwrapper()

def compute_signed_angle(v1, v2):
    """Returns the signed angle in degrees from v1 to v2 (positive = CCW, negative = CW)"""
    angle1 = np.arctan2(v1[1], v1[0])
    angle2 = np.arctan2(v2[1], v2[0])
    angle_rad = angle2 - angle1
    angle_deg = np.degrees(angle_rad-np.pi/2)

    # Normalize to [-180, 180]
    if angle_deg > 180:
        angle_deg -= 360
    elif angle_deg < -180:
        angle_deg += 360

    return angle_deg 

# =====================================================================
# CHECKERBOARD → HOMOGRAPHY (PIXELS → MILLIMETRES ON PLANE)
# =====================================================================
def compute_checkerboard_homography(image_bgr):
    """
    Detects the printed checkerboard in the image and computes a homography
    that maps image pixels to coordinates in millimetres on the checkerboard plane.

    Returns:
        H_img_to_mm: 3x3 homography matrix (image [u,v,1] → [x_mm, y_mm, 1])
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Find inner corners
    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_PATTERN_SIZE)
    if not found:
        raise RuntimeError("Checkerboard not found! Make sure it's fully in view and sharp.")

    # Refine corner positions to sub-pixel accuracy
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001
    )
    corners_refined = cv2.cornerSubPix(
        gray, 
        corners, 
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=criteria
    )

    # Build corresponding object points in mm for the checkerboard plane
    # Coordinate system: (0,0) at one corner, x to the right, y down, in mm
    num_inner_x, num_inner_y = CHECKERBOARD_PATTERN_SIZE
    objp = np.zeros((num_inner_x * num_inner_y, 2), np.float32)
    objp[:, :2] = np.mgrid[0:num_inner_x, 0:num_inner_y].T.reshape(-1, 2)
    objp *= CHECKERBOARD_SQUARE_SIZE_MM  # scale to mm

    # We want image (corners) → real world (objp_mm)
    src_pts = corners_refined.reshape(-1, 2)  # image coordinates
    dst_pts = objp                            # mm coordinates

    H_img_to_mm, mask = cv2.findHomography(src_pts, dst_pts)
    if H_img_to_mm is None:
        raise RuntimeError("Homography computation failed.")

    return H_img_to_mm

def image_points_to_mm(points, H_img_to_mm):
    """
    Transform a list of image points [(u,v), ...] to mm coordinates [(x_mm, y_mm), ...]
    using the given homography.
    """
    pts = np.array(points, dtype=np.float32)
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)])  # (N,3)

    pts_mm_h = (H_img_to_mm @ pts_h.T).T  # (N,3)
    pts_mm = pts_mm_h[:, :2] / pts_mm_h[:, 2:3]
    return [tuple(p) for p in pts_mm]

# =====================================================================
# DETECT RED MARKERS (3 POINTS ALONG THE BEAM)
# =====================================================================
def detect_red_markers(image_bgr, expected_markers=3):
    """
    Detects red blobs in the image and returns their centers, ordered along
    the main axis of the beam.

    Returns:
        ordered_centers: [(base_x, base_y), (joint_x, joint_y), (tip_x, tip_y)]
    """
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    red_ranges = [
        (np.array([0,   120, 80]), np.array([10, 255, 255])),
        (np.array([170, 120, 80]), np.array([180, 255, 255]))
    ]


    red_mask = None
    for lower_red, upper_red in red_ranges:
        temp_mask = cv2.inRange(image_hsv, lower_red, upper_red)
        red_mask = temp_mask if red_mask is None else cv2.bitwise_or(red_mask, temp_mask)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < expected_markers:
        raise ValueError(f"Expected at least {expected_markers} red markers, found {len(contours)}.")

    # Take the largest N contours
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:expected_markers]

    centers = []
    for cnt in sorted_contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
            centers.append((cx, cy))

    if len(centers) < expected_markers:
        raise ValueError("Could not compute all marker centroids.")

    pts = np.array(centers, dtype=np.float32)

    # Order along main beam axis using PCA
    mean = np.mean(pts, axis=0)
    pts_centered = pts - mean
    _, _, vt = np.linalg.svd(pts_centered)
    principal_axis = vt[0]  # direction of beam (unit vector)

    projections = pts_centered @ principal_axis  # scalar coordinate along beam
    order = np.argsort(projections)  # from one end to the other

    ordered_pts = [tuple(pts[i]) for i in order]

    # Convention:
    #   ordered_pts[0] -> base (wire base)
    #   ordered_pts[1] -> joint (wire/magnet interface)
    #   ordered_pts[2] -> tip (magnetic tip)
    return ordered_pts

# =====================================================================
# BEAM + TWO PARTS (WIRE + MAGNETIC TIP)
# =====================================================================
def analyse_beam_two_parts(image_path, show=True):
    """
    1. Detect 3 red markers along the beam.
       - base  (nitinol wire base)
       - joint (wire/magnet interface)
       - tip   (magnetic tip)
    2. Detect checkerboard and compute homography (pixels → mm).
    3. Convert points to mm and compute:
       - wire length (base → joint)
       - magnet length (joint → tip)
       - total length (base → tip)
       - angles for each segment
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    # 1. Detect red markers
    tip_px, joint_px, base_px  = detect_red_markers(image, expected_markers=3)

    # 2. Compute homography from checkerboard
    H_img_to_mm = compute_checkerboard_homography(image)

    (base_mm, joint_mm, tip_mm) = image_points_to_mm(
        [base_px, joint_px, tip_px], 
        H_img_to_mm
    )

    base_mm = np.array(base_mm)
    joint_mm = np.array(joint_mm)
    tip_mm = np.array(tip_mm)

    # --------------------------------------------------------
    # LOCAL COORDINATES: base is (0,0), everything relative to it
    # --------------------------------------------------------
    base_local = np.array([0.0, 0.0])
    joint_local = joint_mm - base_mm
    tip_local   = tip_mm   - base_mm


    # 4. Lengths (px)
    base_px_arr = np.array(base_px)
    joint_px_arr = np.array(joint_px)
    tip_px_arr = np.array(tip_px)

    wire_length_px   = np.linalg.norm(joint_px_arr - base_px_arr)
    magnet_length_px = np.linalg.norm(tip_px_arr - joint_px_arr)
    total_length_px  = np.linalg.norm(tip_px_arr - base_px_arr)

    # 5. Lengths (mm)
    wire_length_mm   = np.linalg.norm(joint_mm - base_mm)
    magnet_length_mm = np.linalg.norm(tip_mm - joint_mm)
    total_length_mm  = np.linalg.norm(tip_mm - base_mm)

    # 6. Angles (using image pixel vectors, just as before)
    reference = np.array([0.0, 1.0])  # "down" in image coordinates

    wire_vec_px   = joint_px_arr - base_px_arr
    magnet_vec_px = tip_px_arr  - joint_px_arr
    full_vec_px   = tip_px_arr  - base_px_arr

    wire_angle   = unwrap_angle(compute_signed_angle(reference, wire_vec_px))
    magnet_angle = unwrap_angle(compute_signed_angle(reference, magnet_vec_px))
    full_angle   = unwrap_angle(compute_signed_angle(reference, full_vec_px))

    if show:
        vis = image.copy()

        # Draw markers
        cv2.circle(vis, tuple(np.int32(base_px_arr)),  7, (0, 255, 0), -1)  # base - green
        cv2.circle(vis, tuple(np.int32(joint_px_arr)), 7, (255, 255, 0), -1)  # joint - cyan
        cv2.circle(vis, tuple(np.int32(tip_px_arr)),   7, (0, 0, 255), -1)  # tip - red

        # Draw segments
        cv2.line(vis, tuple(np.int32(base_px_arr)),  tuple(np.int32(joint_px_arr)), (0, 255, 0), 2)
        cv2.line(vis, tuple(np.int32(joint_px_arr)), tuple(np.int32(tip_px_arr)),   (0, 0, 255), 2)

        text1 = f"Wire: {wire_length_mm:.2f} mm, angle {wire_angle:.1f}°"
        text2 = f"Magnet: {magnet_length_mm:.2f} mm, angle {magnet_angle:.1f}°"
        text3 = f"Total: {total_length_mm:.2f} mm, angle {full_angle:.1f}°"

        cv2.putText(vis, text1, (30, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0),  2, cv2.LINE_AA)
        cv2.putText(vis, text2, (30, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, text3, (30, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title("Beam with wire + magnetic tip segments")
        plt.axis("off")
        plt.show()

    return {
        "base_px": tuple(base_px),
        "joint_px": tuple(joint_px),
        "tip_px": tuple(tip_px),

        "wire_length_px": wire_length_px,
        "magnet_length_px": magnet_length_px,
        "total_length_px": total_length_px,

        "base_mm": tuple(base_mm),
        "joint_mm": tuple(joint_mm),
        "tip_mm": tuple(tip_mm),

        "base_local_mm": tuple(base_local),   # should be (0.0, 0.0)
        "joint_local_mm": tuple(joint_local), # joint in beam frame
        "tip_local_mm": tuple(tip_local),     # tip in beam frame

        "wire_length_mm": wire_length_mm,
        "magnet_length_mm": magnet_length_mm,
        "total_length_mm": total_length_mm,

        "wire_angle_deg": wire_angle,
        "magnet_angle_deg": magnet_angle,
        "total_angle_deg": full_angle
    }


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    img_file = new_capture()
    # Or use your existing path instead:
    # img_file = "/home/jack/literate-code/focused_image.jpg"

    result = analyse_beam_two_parts(img_file, show=True)

    print("Pixel positions:")
    print("  base :", result["base_px"])
    print("  joint:", result["joint_px"])
    print("  tip  :", result["tip_px"])
    print("\nLengths (px):")
    print("  wire   :", result["wire_length_px"])
    print("  magnet :", result["magnet_length_px"])
    print("  total  :", result["total_length_px"])
    print("\nPositions (mm):")
    print("  base :", result["base_mm"])
    print("  joint:", result["joint_mm"])
    print("  tip  :", result["tip_mm"])
    print("\nLengths (mm):")
    print("  wire   :", result["wire_length_mm"])
    print("  magnet :", result["magnet_length_mm"])
    print("  total  :", result["total_length_mm"])
    print("\nAngles (deg):")
    print("  wire   :", result["wire_angle_deg"])
    print("  magnet :", result["magnet_angle_deg"])
    print("  total  :", result["total_angle_deg"])
    print("Base (local):", result["base_local_mm"])
    print("Tip  (local):", result["tip_local_mm"])
