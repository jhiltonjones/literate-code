import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splprep, splev
from image_capture import capture_image
def below_or_above(path):
# Load and convert to grayscale
# image_path = "/home/jack/literate-code/captured_image.jpg"
    image_path = path

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    x_clip_start = 0
    clipped_gray = gray[:, x_clip_start:]
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(clipped_gray)
    base_x = min_loc[0] + x_clip_start  # Corrected to full image coordinate
    base_y = min_loc[1]


    # --- SEARCH BELOW THE BASE FOR THE TRUE TIP ---
    y_start = max(0, base_y - 1)
    y_end = min(gray.shape[0], base_y + 1)

    x_start = max(0, base_x - 100)
    x_end = base_x

    # Extract horizontal strip
    search_strip = gray[y_start:y_end, x_start:x_end]

    # Find all dark pixels below threshold
    dark_threshold = 70
    ys, xs = np.where(search_strip < dark_threshold)

    xs_global = xs + x_start
    ys_global = ys + y_start

    if len(xs_global) > 0:
        leftmost_index = np.argmin(xs_global)  # furthest left (smallest x)
        tip_x = xs_global[leftmost_index]
        tip_y = ys_global[leftmost_index]
        rod_tip_filtered = (tip_x, tip_y)
    else:
        rod_tip_filtered = (base_x, base_y)
    rod_tip_x, rod_tip_y = rod_tip_filtered

    # --- STEP 2: Fit spline through given ring points ---
    ring_coords = np.array([
    (1178.7, 317.4),
    (1043.7, 319.5),
    (929.4, 325.7),
    (877.4, 317.4),
    (802.6, 269.6),
    (692.5, 169.9),
    (605.2, 109.6)
    ])
    x_rings, y_rings = ring_coords[:, 0], ring_coords[:, 1]

    # Fit spline
    tck, _ = splprep([x_rings, y_rings], s=0)
    u_fine = np.linspace(0, 1, 400)
    x_spline, y_spline = splev(u_fine, tck)

    # Find closest spline point to rod tip in full 2D (x, y)
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
    # Draw rings
    for idx, (cx, cy) in enumerate(ring_coords, 1):
        cv2.circle(image_rgb, (int(cx), int(cy)), 15, (255, 0, 0), 3)
        cv2.putText(image_rgb, f'Ring {idx}', (int(cx) - 40, int(cy) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Draw spline
    spline_poly = np.array(list(zip(x_spline, y_spline)), dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(image_rgb, [spline_poly], isClosed=False, color=(0, 255, 0), thickness=2)

    # Draw rod tip
    cv2.circle(image_rgb, rod_tip_filtered, 10, (255, 0, 0), -1)
    cv2.putText(image_rgb, "Rod Tip (darkest)", (rod_tip_x + 10, rod_tip_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Draw closest spline point
    cv2.circle(image_rgb, (int(closest_spline_x), int(closest_spline_y)), 8, (0, 255, 255), -1)
    cv2.putText(image_rgb, "Closest Spline Pt", (int(closest_spline_x) + 10, int(closest_spline_y)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Draw result
    cv2.putText(image_rgb, relation_text, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

    # # Show result
    # plt.figure(figsize=(12, 7))
    # plt.imshow(image_rgb)
    # plt.title("Rod Tip Position vs. Spline")
    # plt.axis('off')
    # plt.show()

    # Print debug info
    print(f"Rod tip detected at: {rod_tip_filtered} with intensity: {min_val}")
    print(relation_text)
    
    return tip
if __name__ == "__main__":
    image = capture_image()
    tip = below_or_above(image)
    if tip == "Below":
        print("WHOO")
