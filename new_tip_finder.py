import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splprep, splev
from new_cam import new_capture

def detect_rod_tip_darkest_right(image_path, graph=True):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_enhanced = cv2.equalizeHist(gray)

    dark_threshold = np.min(gray_enhanced) + 13
    mask_x = (np.arange(gray_enhanced.shape[1]) >= 55) & (np.arange(gray_enhanced.shape[1]) <= 600)
    x_indices = np.where(mask_x)[0]

    # Apply Y range mask (y > 90)
    mask_y = (np.arange(gray_enhanced.shape[0]) > 140) & (np.arange(gray_enhanced.shape[0]) <= 314)
    y_range_indices = np.where(mask_y)[0]

    # Find all pixels below the dark threshold
    y_all, x_all = np.where(gray_enhanced <= dark_threshold)

    # Filter based on x and y indices
    valid_idx = [i for i, (x, y) in enumerate(zip(x_all, y_all)) if x in x_indices and y in y_range_indices]

    if not valid_idx:
        raise ValueError("No dark pixels found within specified x and y ranges")

    x_filtered = x_all[valid_idx]
    y_filtered = y_all[valid_idx]
    rightmost_idx = np.argmax(x_filtered)
    rod_tip = (x_filtered[rightmost_idx], y_filtered[rightmost_idx])
    rod_tip_x, rod_tip_y = rod_tip

    # --- Fit spline through given ring points ---
    ring_coords = np.array([
    (14.1, 240.8),
    (60.1, 244.3),
    (95.0, 253.2),
    (130.7, 268.3),
    (157.4, 285.4),
    (185.6, 292.3),
    (213.0, 283.4),
    (245.9, 260.0),
    (282.3, 225.8),
    (321.4, 196.3),
    (368.0, 184.6),
    (452.3, 186.7),
    (525.7, 189.4),
    (600.5, 191.5),
    (631.3, 194.2)
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
    scale_pixels_per_mm = 2.74
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
