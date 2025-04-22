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

    dark_threshold = np.min(gray_enhanced) + 14
    mask_x = (np.arange(gray_enhanced.shape[1]) >= 95) & (np.arange(gray_enhanced.shape[1]) <= 592)
    x_indices = np.where(mask_x)[0]
    y_indices, x_all = np.where(gray_enhanced <= dark_threshold)
    valid_idx = [i for i, x in enumerate(x_all) if x in x_indices]

    if not valid_idx:
        raise ValueError("No dark pixels found within specified x-range")

    x_filtered = x_all[valid_idx]
    y_filtered = y_indices[valid_idx]
    rightmost_idx = np.argmax(x_filtered)
    rod_tip = (x_filtered[rightmost_idx], y_filtered[rightmost_idx])
    rod_tip_x, rod_tip_y = rod_tip

    # Estimate catheter direction using PCA
    window_size = 20
    x_start = max(0, rod_tip_x - window_size)
    x_end = rod_tip_x
    segment_strip = gray[y_filtered[rightmost_idx]:y_filtered[rightmost_idx]+1, x_start:x_end]
    ys, xs = np.where(segment_strip < dark_threshold)
    if len(xs) >= 2:
        points = np.column_stack((xs + x_start, ys + y_filtered[rightmost_idx]))
        mean = np.mean(points, axis=0)
        cov = np.cov(points - mean, rowvar=False)
        eigvals, eigvecs = np.linalg.eig(cov)
        catheter_dir = eigvecs[:, np.argmax(eigvals)]
    else:
        mean = np.array([rod_tip_x, rod_tip_y])
        catheter_dir = np.array([1, 0])

    ring_coords = np.array([
        (86.3, 248.9), (137.2, 258.2), (182.9, 269.6), (218.2, 283.1),
        (251.4, 308.1), (280.5, 317.4), (307.6, 319.5), (348.1, 303.9),
        (390.7, 267.6), (444.7, 222.9), (501.8, 196.9), (571.4, 203.1), (620.3, 205.2)
    ])
    tck, _ = splprep(ring_coords.T, s=0)
    u_fine = np.linspace(0, 1, 400)
    x_spline, y_spline = splev(u_fine, tck)

    spline_points = np.array(list(zip(x_spline, y_spline)))
    tip_point = np.array([rod_tip_x, rod_tip_y])
    distances = np.linalg.norm(spline_points - tip_point, axis=1)
    closest_index = np.argmin(distances)
    closest_spline_x, closest_spline_y = spline_points[closest_index]
    scale_pixels_per_mm = 9.55

    spline_coords = np.array([x_spline, y_spline]).T
    deltas = np.diff(spline_coords, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    arc_lengths = np.insert(np.cumsum(segment_lengths), 0, 0)
    closest_arc_length = arc_lengths[closest_index]
    target_arc_length = closest_arc_length + 10 * scale_pixels_per_mm

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
        if tip == "Below":
            alignment_angle *= -1

    if graph:
        cv2.circle(image_rgb, (int(target_point[0]), int(target_point[1])), 6, (0, 0, 255), -1)
        cv2.putText(image_rgb, "10mm along spline", (int(target_point[0]) + 10, int(target_point[1])),
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
