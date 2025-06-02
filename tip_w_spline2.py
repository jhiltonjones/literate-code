import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splprep, splev
from image_capture import capture_image

def below_or_above(path):
    # Load and convert to grayscale
    image_path = path
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to create a binary mask of the rod (assuming it's dark)
    dark_threshold = 80
    _, rod_mask = cv2.threshold(gray, dark_threshold, 255, cv2.THRESH_BINARY_INV)

    # Find contours of the rod
    contours, _ = cv2.findContours(rod_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Combine all points from all contours into one array
        all_points = np.vstack(contours).squeeze()

        # Find the leftmost point (smallest x)
        leftmost_idx = np.argmin(all_points[:, 0])
        tip_x, tip_y = all_points[leftmost_idx]
        rod_tip_filtered = (int(tip_x), int(tip_y))
    else:
        print("No rod detected in mask. Falling back to default tip.")
        # Fallback to some point if contour detection fails
        tip_x, tip_y = 800, 300  # arbitrary fallback
        rod_tip_filtered = (tip_x, tip_y)

    rod_tip_x, rod_tip_y = rod_tip_filtered

    # --- STEP 2: Fit spline through given ring points ---
    ring_coords = np.array([
    (593.8, 183.2),
    (561.6, 183.2),
    (529.5, 177.5),
    (510.2, 172.5),
    (484.5, 158.9),
    (466.7, 151.8),
    (452.4, 151.8),
    (427.4, 158.2),
    (411.7, 171.0),
    (390.2, 191.0),
    (371.7, 205.3),
    (341.0, 225.3),
    (303.8, 228.9),
    (244.6, 236.0),
    (193.9, 235.3),
    (164.6, 225.3),
    (155.3, 206.0),
    (153.2, 176.0),
    (144.6, 141.8)
    ])
    x_rings, y_rings = ring_coords[:, 0], ring_coords[:, 1]

    tck, _ = splprep([x_rings, y_rings], s=0)
    u_fine = np.linspace(0, 1, 400)
    x_spline, y_spline = splev(u_fine, tck)

    spline_points = np.array(list(zip(x_spline, y_spline)))
    tip_point = np.array([rod_tip_x, rod_tip_y])
    distances = np.linalg.norm(spline_points - tip_point, axis=1)
    closest_index = np.argmin(distances)
    closest_spline_x, closest_spline_y = spline_points[closest_index]

    # --- STEP 3: Determine position relative to spline ---
    if abs(rod_tip_y - closest_spline_y) <= 0.02:
        relation_text = "Rod tip is aligned with spline (no adjustment needed)"
        tip = "inline"
    elif rod_tip_y < closest_spline_y:
        relation_text = "Rod tip is ABOVE the spline"
        tip = "Above"
    else:
        relation_text = "Rod tip is BELOW the spline"
        tip = "Below"

    # --- STEP 4: Annotate on image ---
    for idx, (cx, cy) in enumerate(ring_coords, 1):
        cv2.circle(image_rgb, (int(cx), int(cy)), 15, (255, 0, 0), 3)
        cv2.putText(image_rgb, f'Ring {idx}', (int(cx) - 40, int(cy) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    spline_poly = np.array(list(zip(x_spline, y_spline)), dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(image_rgb, [spline_poly], isClosed=False, color=(0, 255, 0), thickness=2)

    cv2.circle(image_rgb, rod_tip_filtered, 10, (255, 0, 0), -1)
    cv2.putText(image_rgb, "Rod Tip (leftmost)", (rod_tip_x + 10, rod_tip_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.circle(image_rgb, (int(closest_spline_x), int(closest_spline_y)), 8, (0, 255, 255), -1)
    cv2.putText(image_rgb, "Closest Spline Pt", (int(closest_spline_x) + 10, int(closest_spline_y)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.putText(image_rgb, relation_text, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

    plt.figure(figsize=(12, 7))
    plt.imshow(image_rgb)
    plt.title("Rod Tip Position vs. Spline")
    plt.axis('off')
    plt.show()

    print(f"Rod tip detected at: {rod_tip_filtered}")
    print(relation_text)

    return tip

if __name__ == "__main__":
    image = capture_image()
    tip = below_or_above(image)
    if tip == "Below":
        print("WHOO")
