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

    dark_threshold = np.min(gray_enhanced) + 10
    mask_x = (np.arange(gray_enhanced.shape[1]) >= 140) & (np.arange(gray_enhanced.shape[1]) <= 600)
    x_indices = np.where(mask_x)[0]
    # exclude_x_min, exclude_x_max = 250, 383
    # exclude_y_min, exclude_y_max = 185, 218
    # Apply Y range mask (y > 90)
    mask_y = (np.arange(gray_enhanced.shape[0]) > 83) & (np.arange(gray_enhanced.shape[0]) <= 242)
    y_range_indices = np.where(mask_y)[0]

    # Find all pixels below the dark threshold
    y_all, x_all = np.where(gray_enhanced <= dark_threshold)

    # Define exclusion zone
    exclude_x_min, exclude_x_max = 0,0
    exclude_y_min, exclude_y_max = 0,0

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
    leftmost_idx = np.argmin(x_filtered)

    rod_tip = (x_filtered[leftmost_idx], y_filtered[leftmost_idx])
    rod_tip_x, rod_tip_y = rod_tip
    # Step 1: Look 30 pixels to the left
    search_x = rod_tip_x - 10
    if search_x < 0:
        raise ValueError("Cannot look 30 pixels to the left — out of image bounds.")

    # Step 2: Define vertical search range (e.g., ±5 pixels)
    y_min = max(rod_tip_y - 5, 0)
    y_max = min(rod_tip_y + 5, gray_enhanced.shape[0])

    # Step 3: Extract column of interest
    search_column = gray_enhanced[y_min:y_max, search_x]

    # Step 4: Find the darkest pixel in this column
    local_min_idx = np.argmin(search_column)
    next_darkest_y = y_min + local_min_idx
    next_darkest_tip = (search_x, next_darkest_y)

    # Update rod_tip to the new detected point
    rod_tip = next_darkest_tip
    rod_tip_x, rod_tip_y = rod_tip

    # --- Fit spline through given ring points ---
    ring_coords = np.array([
    (608.2, 154.4),
    (577.7, 151.0),
    (552.6, 142.2),
    (531.6, 134.7),
    (504.5, 122.5),
    (480.1, 118.5),
    (466.6, 118.5),
    (438.1, 131.4),
    (415.7, 149.0),
    (388.0, 159.8),
    (358.1, 176.1),
    (315.5, 188.3),
    (262.6, 199.1),
    (207.0, 195.7),
    (175.9, 186.9),
    (158.3, 162.5),
    (150.1, 131.4),
    (154.9, 94.1),
    (167.7, 59.5)
    # (488.9, 144.0),
    # (435.4, 137.2),
    # (418.5, 127.1),
    # (404.3, 117.6),
    # (385.3, 108.1),
    # (369.1, 108.8),
    # (345.4, 123.0),
    # (325.1, 129.8),
    # (300.7, 154.1)
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
    scale_pixels_per_mm = 2.72
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
