import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splprep, splev
from new_cam import new_capture

def detect_rod_tip_darkest_right(image_path, graph=True):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhance contrast for clearer detection
    gray_enhanced = cv2.equalizeHist(gray)

    # Relax dark pixel threshold to capture the tip accurately
    dark_threshold = np.min(gray_enhanced) + 14
    dark_pixels = np.where((gray_enhanced <= dark_threshold) & (np.arange(gray_enhanced.shape[1])[None, :] >= 95) & (np.arange(gray_enhanced.shape[1])[None, :] <= 592))

    # Select the rightmost dark pixel within specified x-range
    if len(dark_pixels[1]) == 0:
        raise ValueError("No dark pixels found within specified x-range")

    rightmost_idx = np.argmax(dark_pixels[1])
    rod_tip = (dark_pixels[1][rightmost_idx], dark_pixels[0][rightmost_idx])

    rod_tip_x, rod_tip_y = rod_tip

    # --- Fit spline through given ring points ---
    ring_coords = np.array([
    (86.3, 248.9),
    (137.2, 258.2),
    (182.9, 269.6),
    (218.2, 283.1),
    (251.4, 308.1),
    (280.5, 317.4),
    (307.6, 319.5),
    (348.1, 303.9),
    (390.7, 267.6),
    (444.7, 222.9),
    (501.8, 196.9),
    (571.4, 203.1),
    (620.3, 205.2)
    ])
    tck, _ = splprep(ring_coords.T, s=0)
    u_fine = np.linspace(0, 1, 400)
    spline_coords = np.array(splev(u_fine, tck)).T

    # Closest spline point
    distances = np.linalg.norm(spline_coords - rod_tip, axis=1)
    closest_idx = np.argmin(distances)
    closest_spline_pt = spline_coords[closest_idx]

    # Signed distance calculation
    pixel_distance = distances[closest_idx]
    scale_pixels_per_mm = 3.2
    distance_mm = pixel_distance / scale_pixels_per_mm
    signed_distance_mm = distance_mm if rod_tip_y < closest_spline_pt[1] else -distance_mm

    relation_text = "Rod tip aligned"
    if abs(rod_tip_y - closest_spline_pt[1]) > 2:
        relation_text = "Rod tip ABOVE spline" if rod_tip_y < closest_spline_pt[1] else "Rod tip BELOW spline"

    # Visual annotations
    cv2.polylines(image_rgb, [spline_coords.astype(int)], False, (0,255,0), 2)
    cv2.circle(image_rgb, rod_tip, 8, (255,0,0), -1)
    cv2.circle(image_rgb, tuple(closest_spline_pt.astype(int)), 8, (0,255,255), -1)
    cv2.putText(image_rgb, relation_text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    if graph:
        plt.figure(figsize=(12,7))
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.title('Rod Tip Detection (Darkest Rightmost Pixel)')
        plt.show()

    print(f"Signed distance from rod tip to spline: {signed_distance_mm:.2f} mm")
    print(relation_text)

    return relation_text, rod_tip, signed_distance_mm, closest_spline_pt


if __name__ == "__main__":
    image_path = new_capture()
    relation, tip_pos, dist_mm, spline_pt = detect_rod_tip_darkest_right(image_path, graph=True)
    print("Tip position:", tip_pos)
    print("Distance (mm):", dist_mm)
