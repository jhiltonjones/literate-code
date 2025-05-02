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

    dark_threshold = np.min(gray_enhanced) + 18
    mask_x = (np.arange(gray_enhanced.shape[1]) >= 55) & (np.arange(gray_enhanced.shape[1]) <= 600)
    x_indices = np.where(mask_x)[0]

    # Apply Y range mask (y > 90)
    mask_y = (np.arange(gray_enhanced.shape[0]) > 140) & (np.arange(gray_enhanced.shape[0]) <= 314)
    y_range_indices = np.where(mask_y)[0]

    # Find all pixels below the dark threshold
    y_all, x_all = np.where(gray_enhanced <= dark_threshold)

    # Define exclusion zone
    exclude_x_min, exclude_x_max = 453, 478
    exclude_y_min, exclude_y_max = 157, 165

    # Filter based on x/y range and exclusion box
    valid_idx = [
        i for i, (x, y) in enumerate(zip(x_all, y_all))
        if (x in x_indices and y in y_range_indices) and not (
            exclude_x_min <= x <= exclude_x_max and exclude_y_min <= y <= exclude_y_max
        )
    ]


    if not valid_idx:
        raise ValueError("No dark pixels found within specified x and y ranges")

    x_filtered = x_all[valid_idx]
    y_filtered = y_all[valid_idx]
    rightmost_idx = np.argmax(x_filtered)
    rod_tip = (x_filtered[rightmost_idx], y_filtered[rightmost_idx])
    rod_tip_x, rod_tip_y = rod_tip

    # --- Fit spline through given ring points ---
    ring_coords = np.array([
    (24.2, 232.1),
    (62.9, 238.7),
    (99.0, 245.3),
    (133.9, 252.5),
    (151.6, 266.3),
    (173.3, 274.8),
    (193.0, 277.4),
    (225.2, 268.2),
    (256.7, 237.4),
    (280.3, 214.4),
    (314.5, 190.1),
    (357.8, 177.6),
    (415.0, 181.5),
    (476.1, 181.5),
    (579.2, 190.1),
    (623.2, 191.4)
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
