import cv2
import numpy as np
import glob
import os

# === CONFIG ===

# Your checkerboard: 9x6 squares -> 8x5 inner corners
PATTERN_COLS = 8  # inner corners across (cols)
PATTERN_ROWS = 5  # inner corners down (rows)
SQUARE_SIZE_MM = 10.0

IMG_DIR = "/home/jack/literate-code/calib_images/*.jpg"
OUT_FILE = "/home/jack/literate-code/camera_params.npz"

# === PREP OBJECT POINTS (3D points on the board) ===

objp = np.zeros((PATTERN_ROWS * PATTERN_COLS, 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN_COLS, 0:PATTERN_ROWS].T.reshape(-1, 2)
objp *= SQUARE_SIZE_MM  # units: mm

objpoints = []  # 3D points in world space (board)
imgpoints = []  # 2D points in image

images = glob.glob(IMG_DIR)
print(f"Found {len(images)} images")

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Could not read {fname}, skipping.")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    pattern_size = (PATTERN_COLS, PATTERN_ROWS)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags=flags)

    if ret:
        # refine corner locations
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints.append(corners)

        # show detection
        cv2.drawChessboardCorners(img, pattern_size, corners, ret)
        cv2.imshow("Calibration - detected corners", img)
        cv2.waitKey(100)
    else:
        print(f"Checkerboard NOT found in {fname}")

cv2.destroyAllWindows()

if len(objpoints) < 5:
    print("Not enough valid images for calibration (need ~10+ with detected corners).")
    exit(1)

# === CALIBRATE ===

img_shape = gray.shape[::-1]  # (width, height) from last image
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_shape, None, None
)

print("Calibration RMS reprojection error:", ret)
print("Camera matrix:\n", camera_matrix)
print("Distortion coefficients:\n", dist_coeffs.ravel())

# Save
np.savez(OUT_FILE, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
print(f"Saved camera parameters to {OUT_FILE}")
