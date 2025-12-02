import cv2
import numpy as np

# =============================================
# CONFIGURATION
# =============================================

# --- Physical checkerboard parameters (MUST match what you printed) ---

# Square size in millimetres (same as in your generator)
SQUARE_SIZE_MM = 10.0  

# Base checkerboard: 9 x 6 squares  -> inner corners: 8 x 5
BASE_SQUARES_COLS = 9
BASE_SQUARES_ROWS = 6
BASE_PATTERN_COLS = BASE_SQUARES_COLS - 1   # inner corners
BASE_PATTERN_ROWS = BASE_SQUARES_ROWS - 1

# End-effector checkerboard: e.g. 7 x 5 squares -> inner corners: 6 x 4
EE_SQUARES_COLS = 7
EE_SQUARES_ROWS = 5
EE_PATTERN_COLS = EE_SQUARES_COLS - 1
EE_PATTERN_ROWS = EE_SQUARES_ROWS - 1

# Path to camera intrinsics
CALIB_FILE = "/home/jack/literate-code/camera_params.npz"

# =============================================
# HELPER FUNCTIONS
# =============================================

def load_camera_params(filename):
    data = np.load(filename, allow_pickle=True)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]
    return camera_matrix, dist_coeffs



def create_object_points(pattern_cols, pattern_rows, square_size_mm):
    """
    Create a grid of 3D points for the checkerboard corners in its own coordinate frame.
    Z = 0 (flat board).
    """
    objp = np.zeros((pattern_rows * pattern_cols, 3), np.float32)
    # Note: pattern_cols is along x, pattern_rows along y
    objp[:, :2] = np.mgrid[0:pattern_cols, 0:pattern_rows].T.reshape(-1, 2)
    objp *= square_size_mm  # Keep units in millimetres
    return objp


def find_checkerboard_pose(gray, pattern_cols, pattern_rows, obj_points, camera_matrix, dist_coeffs):
    """
    Find chessboard corners and estimate pose using solvePnP.
    Returns (retval, rvec, tvec, corners).
    retval is True if found, False otherwise.
    """
    pattern_size = (pattern_cols, pattern_rows)

    # Try to find chessboard corners
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags=flags)

    if not found:
        return False, None, None, None

    # Refine corner locations
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Solve PnP: object points (3D) to image points (2D)
    success, rvec, tvec = cv2.solvePnP(
        obj_points, 
        corners, 
        camera_matrix, 
        dist_coeffs, 
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return False, None, None, None

    return True, rvec, tvec, corners


def draw_axes(img, camera_matrix, dist_coeffs, rvec, tvec, length=50.0):
    """
    Draw XYZ axes on the image for visualization.
    length: in same units as object points (mm).
    """
    axis = np.float32([
        [0, 0, 0],
        [length, 0, 0],   # X - red
        [0, length, 0],   # Y - green
        [0, 0, -length]   # Z - blue (towards camera is -Z in many setups)
    ])

    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = imgpts.reshape(-1, 2).astype(int)

    origin = tuple(imgpts[0])
    x_axis = tuple(imgpts[1])
    y_axis = tuple(imgpts[2])
    z_axis = tuple(imgpts[3])

    cv2.line(img, origin, x_axis, (0, 0, 255), 2)
    cv2.line(img, origin, y_axis, (0, 255, 0), 2)
    cv2.line(img, origin, z_axis, (255, 0, 0), 2)


# =============================================
# MAIN: capture one frame and compute distance
# =============================================

def main():
    # Load camera intrinsics
    camera_matrix, dist_coeffs = load_camera_params(CALIB_FILE)

    # Precompute object points for both boards
    objp_base = create_object_points(BASE_PATTERN_COLS, BASE_PATTERN_ROWS, SQUARE_SIZE_MM)
    objp_ee   = create_object_points(EE_PATTERN_COLS,   EE_PATTERN_ROWS,   SQUARE_SIZE_MM)

    # Open default camera (change index if needed)
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press 'q' to quit, 'm' to measure on current frame.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Undistort for better detection (optional but nice)
        h, w = frame.shape[:2]
        new_cam_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_cam_mtx)

        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

        # Try find base checkerboard
        base_found, base_rvec, base_tvec, base_corners = find_checkerboard_pose(
            gray, BASE_PATTERN_COLS, BASE_PATTERN_ROWS, objp_base, camera_matrix, dist_coeffs
        )

        # Try find end-effector checkerboard
        ee_found, ee_rvec, ee_tvec, ee_corners = find_checkerboard_pose(
            gray, EE_PATTERN_COLS, EE_PATTERN_ROWS, objp_ee, camera_matrix, dist_coeffs
        )

        display = undistorted.copy()

        if base_found:
            cv2.drawChessboardCorners(display, (BASE_PATTERN_COLS, BASE_PATTERN_ROWS), base_corners, base_found)
            draw_axes(display, camera_matrix, dist_coeffs, base_rvec, base_tvec)
            cv2.putText(display, "Base board", tuple(base_corners[0,0].astype(int) + np.array([0, -10])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if ee_found:
            cv2.drawChessboardCorners(display, (EE_PATTERN_COLS, EE_PATTERN_ROWS), ee_corners, ee_found)
            draw_axes(display, camera_matrix, dist_coeffs, ee_rvec, ee_tvec)
            cv2.putText(display, "EE board", tuple(ee_corners[0,0].astype(int) + np.array([0, -10])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # If both found, compute distance between their origins in camera coordinates
        if base_found and ee_found:
            # tvecs are positions of each board's origin in camera frame, in mm
            base_pos = base_tvec.reshape(3)
            ee_pos   = ee_tvec.reshape(3)

            diff = ee_pos - base_pos
            distance_mm = np.linalg.norm(diff)
            distance_str = f"Distance: {distance_mm/10:.1f} mm"

            cv2.putText(display, distance_str, (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        else:
            cv2.putText(display, "Looking for both checkerboards...", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Checkerboard pose & distance", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
