import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splprep, splev
from new_cam import new_capture

def compute_angle(v1, v2):
    """Returns the angle in degrees between vectors 'v1' and 'v2'"""
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_prod = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
    angle_rad = np.arccos(dot_prod)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def detect_rod_tip_darkest_right(image_path, graph=True):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_enhanced = cv2.equalizeHist(gray)

    dark_threshold = np.min(gray_enhanced) + 12
    mask_x = (np.arange(gray_enhanced.shape[1]) >= 30) & (np.arange(gray_enhanced.shape[1]) <= 600)
    x_indices = np.where(mask_x)[0]

    # Apply Y range mask (y > 90)
    mask_y = (np.arange(gray_enhanced.shape[0]) > 140) & (np.arange(gray_enhanced.shape[0]) <= 320)
    y_range_indices = np.where(mask_y)[0]

    # Find all pixels below the dark threshold
    y_all, x_all = np.where(gray_enhanced <= dark_threshold)

    # Filter based on x and y indices
    # Define exclusion zone
    exclude_x_min, exclude_x_max = 452, 478
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

    # Estimate catheter direction using PCA
    # Estimate direction by collecting dark pixels in a local leftward region
    window_x = 30  # search 30 pixels to the left
    window_y = 10  # search 10 pixels above and below

    x_min = max(0, rod_tip_x - window_x)
    x_max = rod_tip_x
    y_min = max(0, rod_tip_y - window_y)
    y_max = min(gray_enhanced.shape[0], rod_tip_y + window_y)

    local_region = gray_enhanced[y_min:y_max, x_min:x_max]
    ys_local, xs_local = np.where(local_region <= dark_threshold)

    if len(xs_local) >= 2:
        xs_global = xs_local + x_min
        ys_global = ys_local + y_min
        points = np.column_stack((xs_global, ys_global))
        mean = np.mean(points, axis=0)
        cov = np.cov(points - mean, rowvar=False)
        eigvals, eigvecs = np.linalg.eig(cov)
        principal_direction = eigvecs[:, np.argmax(eigvals)]

        # Ensure the direction points rightward (toward rod tip)
        if np.dot(principal_direction, [1, 0]) < 0:
            principal_direction *= -1

        catheter_dir = principal_direction
    else:
        mean = np.array([rod_tip_x, rod_tip_y])
        catheter_dir = np.array([1, 0])


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
    x_spline, y_spline = splev(u_fine, tck)

    spline_points = np.array(list(zip(x_spline, y_spline)))
    tip_point = np.array([rod_tip_x, rod_tip_y])
    distances = np.linalg.norm(spline_points - tip_point, axis=1)
    closest_index = np.argmin(distances)
    closest_spline_x, closest_spline_y = spline_points[closest_index]
    scale_pixels_per_mm = 2.74

    spline_coords = np.array([x_spline, y_spline]).T
    deltas = np.diff(spline_coords, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    arc_lengths = np.insert(np.cumsum(segment_lengths), 0, 0)
    closest_arc_length = arc_lengths[closest_index]
    target_arc_length = closest_arc_length + 20 * scale_pixels_per_mm

    if target_arc_length >= arc_lengths[-1]:
        new_index = len(spline_coords) - 1
    else:
        new_index = np.searchsorted(arc_lengths, target_arc_length)

    target_point = spline_coords[new_index]

    if 0 < new_index < len(spline_coords) - 1:
        prev = spline_coords[new_index - 1]
        nxt = spline_coords[new_index + 1]
        tangent_new = nxt - prev
    else:
        tangent_new = np.array([1, 0])

    unit_tangent_new = tangent_new / np.linalg.norm(tangent_new)
    alignment_angle = compute_angle(catheter_dir, unit_tangent_new)
    alignment_angle = min(alignment_angle, 180 - alignment_angle)

    rod_tip_filtered = rod_tip
    pixel_distance = np.linalg.norm(np.array(rod_tip_filtered) - np.array([closest_spline_x, closest_spline_y]))
    desired_point = np.array([closest_spline_x, closest_spline_y])
    distance_mm = pixel_distance / scale_pixels_per_mm

    if rod_tip_y < closest_spline_y:
        signed_distance_mm = +distance_mm
    elif rod_tip_y > closest_spline_y:
        signed_distance_mm = -distance_mm
    else:
        signed_distance_mm = 0.0

    relation_text = "Rod tip is aligned with spline (no adjustment needed)"
    tip = "inline"
    if abs(rod_tip_y - closest_spline_y) > 2:
        relation_text = "Rod tip ABOVE spline" if rod_tip_y < closest_spline_y else "Rod tip BELOW spline"
        tip = "Above" if rod_tip_y < closest_spline_y else "Below"
        if tip == "Above":
            alignment_angle *= -1

    if graph:
        cv2.circle(image_rgb, (int(target_point[0]), int(target_point[1])), 6, (0, 0, 255), -1)
        cv2.putText(image_rgb, "20mm along spline", (int(target_point[0]) + 10, int(target_point[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        spline_poly = np.array(list(zip(x_spline, y_spline)), dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image_rgb, [spline_poly], isClosed=False, color=(0, 255, 0), thickness=2)
        cv2.circle(image_rgb, rod_tip, 10, (255, 0, 0), -1)
        cv2.putText(image_rgb, "Rod Tip (darkest)", (rod_tip_x + 10, rod_tip_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.circle(image_rgb, (int(closest_spline_x), int(closest_spline_y)), 8, (0, 255, 255), -1)
        cv2.putText(image_rgb, "Closest Spline Pt", (int(closest_spline_x) + 10, int(closest_spline_y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(image_rgb, relation_text, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
        pt1 = (int(mean[0]), int(mean[1]))
        pt2 = (int(mean[0] + 30 * catheter_dir[0]), int(mean[1] + 30 * catheter_dir[1]))
        cv2.arrowedLine(image_rgb, pt1, pt2, (0, 0, 255), 2)

        plt.figure(figsize=(12, 7))
        plt.imshow(image_rgb)
        plt.title("Rod Tip Position vs. Spline")
        plt.axis('off')
        plt.draw()
        # plt.pause(5.4)
        # plt.close()
        plt.show()

    print(f"Signed distance from rod tip to spline: {signed_distance_mm:.2f} mm")
    print(f"Alignment angle between catheter and spline direction: {alignment_angle:.2f} degrees")
    print(relation_text)

    return tip, rod_tip, signed_distance_mm, desired_point, alignment_angle


if __name__ == "__main__":
    image_path = new_capture()
    relation, tip_pos, dist_mm, spline_pt, angle = detect_rod_tip_darkest_right(image_path, graph=True)
    print("Tip position:", tip_pos)
    print("Distance (mm):", dist_mm)
    print("Angle (deg):", angle)
