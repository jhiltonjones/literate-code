import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splprep, splev
from image_capture import capture_image
from new_cam import new_capture
def below_or_above(path, graph):
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
    # Compute Euclidean distance between rod tip and closest spline point
    pixel_distance = np.linalg.norm(np.array(rod_tip_filtered) - np.array([closest_spline_x, closest_spline_y]))
    desired_point = np.array([closest_spline_x, closest_spline_y])
    # Convert pixels to mm
    scale_pixels_per_mm = 9.55
    distance_mm = pixel_distance / scale_pixels_per_mm

    # print(f"Distance from rod tip to spline: {distance_mm:.2f} mm")
    if rod_tip_y < closest_spline_y:
        signed_distance_mm = +distance_mm
    elif rod_tip_y > closest_spline_y:
        signed_distance_mm = -distance_mm
    else:
        signed_distance_mm = 0.0
    print(f"Signed distance from rod tip to spline: {signed_distance_mm:.2f} mm")
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
    # for idx, (cx, cy) in enumerate(ring_coords, 1):
    #     cv2.circle(image_rgb, (int(cx), int(cy)), 15, (255, 0, 0), 3)
    #     cv2.putText(image_rgb, f'Ring {idx}', (int(cx) - 40, int(cy) - 20),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

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
    # --- NEW STEP: Split spline into 50mm increments ---
    scale_pixels_per_mm = 9.55
    distance_mm = 10
    distance_pixels = distance_mm * scale_pixels_per_mm

    # Calculate cumulative arc length of spline
    spline_coords = np.array([x_spline, y_spline]).T
    deltas = np.diff(spline_coords, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    arc_lengths = np.cumsum(segment_lengths)
    arc_lengths = np.insert(arc_lengths, 0, 0)

    # Determine indices at approx 50mm (477.5 px) intervals
    num_segments = int(arc_lengths[-1] // distance_pixels)
    milestone_points = []

    for i in range(1, num_segments + 1):
        target_length = i * distance_pixels
        idx = np.searchsorted(arc_lengths, target_length)
        if idx < len(spline_coords):
            milestone_points.append(spline_coords[idx])

    # Draw milestone points on image
    for i, (mx, my) in enumerate(milestone_points, 1):
        cv2.circle(image_rgb, (int(mx), int(my)), 6, (255, 255, 0), -1)
        cv2.putText(image_rgb, f"{i * distance_mm}mm", (int(mx) + 5, int(my) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    # --- NEW STEP: Compute angles between consecutive segments ---
    def compute_angle(v1, v2):
        """Returns the angle in degrees between vectors 'v1' and 'v2'"""
        unit_v1 = v1 / np.linalg.norm(v1)
        unit_v2 = v2 / np.linalg.norm(v2)
        dot_prod = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)  # clip to avoid numerical issues
        angle_rad = np.arccos(dot_prod)
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    # Compute and annotate angles between segments
    for i in range(1, len(milestone_points) - 1):
        pt_before = milestone_points[i - 1]
        pt_current = milestone_points[i]
        pt_after = milestone_points[i + 1]

        vec1 = pt_current - pt_before
        vec2 = pt_after - pt_current

        angle = compute_angle(vec1, vec2)
        
        # Draw the angle value near the current point
        text = f"{angle:.1f}°"
        cv2.putText(image_rgb, text, (int(pt_current[0]) + 10, int(pt_current[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
    if graph == True:
        # # Show result
        plt.figure(figsize=(12, 7))
        plt.imshow(image_rgb)
        plt.title("Rod Tip Position vs. Spline")
        plt.axis('off')
        plt.show()

    # Print debug info
    # print(f"Rod tip detected at: {rod_tip_filtered} with intensity: {min_val}")
    print(relation_text)
    
    return tip, rod_tip_filtered, signed_distance_mm, desired_point
if __name__ == "__main__":
    # image = capture_image()
    image = new_capture()
    tip, rod_pos, signed_distance_mm, desired_point = below_or_above(image, True)
    if tip == "Below":
        print("WHOO")
    print(rod_pos)
    print(signed_distance_mm, desired_point)