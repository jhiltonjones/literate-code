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
import numpy as np
import cv2

def estimate_catheter_direction(gray_enhanced, rod_tip_x, rod_tip_y, dark_threshold, debug_image=None):
    window_x = 30
    window_y = 10

    x_min = rod_tip_x
    x_max = min(gray_enhanced.shape[1], rod_tip_x + window_x)
    y_min = max(0, rod_tip_y - window_y)
    y_max = min(gray_enhanced.shape[0], rod_tip_y + window_y)

    local_region = gray_enhanced[y_min:y_max, x_min:x_max]
    ys_local, xs_local = np.where(local_region <= dark_threshold)

    if len(xs_local) >= 2:
        xs_global = xs_local + x_min
        ys_global = ys_local + y_min
        points = np.column_stack((xs_global, ys_global))

        centered = points - np.mean(points, axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        principal_direction = vh[0]

        to_tip = np.array([rod_tip_x, rod_tip_y]) - np.mean(points, axis=0)
        if np.dot(principal_direction, to_tip) < 0:
            principal_direction *= -1

        catheter_dir = principal_direction / np.linalg.norm(principal_direction)

        if debug_image is not None:
            pt1 = (int(rod_tip_x), int(rod_tip_y))
            pt2 = (int(rod_tip_x - 20 * catheter_dir[0]), int(rod_tip_y - 20 * catheter_dir[1]))
            cv2.arrowedLine(debug_image, pt1, pt2, (0, 255, 0), 2, tipLength=0.3)

        return catheter_dir, np.mean(points, axis=0)

    else:
        print("Warning: Not enough dark pixels found. Using fallback direction.")
        catheter_dir = np.array([1.0, 0.0])
        mean_point = np.array([rod_tip_x, rod_tip_y])
        return catheter_dir, mean_point



def detect_rod_tip_yellow_right(image_path, graph=False):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_x = (np.arange(image.shape[1]) >= 37) & (np.arange(image.shape[1]) <= 415)
    x_indices = np.where(mask_x)[0]
    mask_y = (np.arange(image.shape[0]) > 65) & (np.arange(image.shape[0]) <= 230)
    y_indices = np.where(mask_y)[0]

    red_ranges = [
        (np.array([0, 50, 50]), np.array([10, 255, 255])),     # lower reds
        (np.array([160, 50, 50]), np.array([180, 255, 255]))   # upper reds
    ]

    red_mask = None
    largest_contour = None

    for lower_red, upper_red in red_ranges:
        temp_mask = cv2.inRange(image_hsv, lower_red, upper_red)
        region_mask = np.zeros_like(temp_mask)
        region_mask[np.ix_(y_indices, x_indices)] = 1
        masked_red = cv2.bitwise_and(temp_mask, temp_mask, mask=region_mask.astype(np.uint8))
        contours, _ = cv2.findContours(masked_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            red_mask = masked_red
            break

    if largest_contour is None:
        raise ValueError("No red tip found in any HSV range.")

    rod_tip_point = min(largest_contour[:, 0, :], key=lambda p: p[0])

    rod_tip_x, rod_tip_y = int(rod_tip_point[0]), int(rod_tip_point[1])
    rod_tip = (rod_tip_x, rod_tip_y)

    contour_points = largest_contour[:, 0, :]
    sorted_points = contour_points[np.argsort(-contour_points[:, 0])]
    N = max(10, len(sorted_points) // 3)
    base_points = sorted_points[:N]

# Use all contour points to estimate the shaft direction
    mean = np.mean(contour_points, axis=0)
    centered = contour_points - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    principal_direction = vh[0]

    # Flip direction if it's pointing away from the tip
    tip_vector = np.array([rod_tip_x, rod_tip_y]) - mean
    if np.dot(principal_direction, tip_vector) > 0:
        principal_direction *= -1

    catheter_dir = principal_direction / np.linalg.norm(principal_direction)


    if graph:
        pt1 = tuple(np.round(mean).astype(int))
        pt2 = tuple(np.round(mean + 40 * catheter_dir).astype(int))
        cv2.arrowedLine(image_rgb, pt1, pt2, (255, 0, 255), 2, tipLength=0.1)

        cv2.circle(image_rgb, (rod_tip_x, rod_tip_y), 5, (0, 255, 0), -1)
        cv2.imshow("Detected Rod Tip and Direction", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()




    # # --- Estimate direction using grayscale ---
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray_enhanced = cv2.equalizeHist(gray)
    # dark_threshold = np.min(gray_enhanced) + 15

    # catheter_dir, mean = estimate_catheter_direction(
    #     gray_enhanced, rod_tip_x, rod_tip_y, dark_threshold, debug_image=image_rgb
    # )

    ring_coords = np.array([
    (391.0, 157.3),
    (354.1, 154.1),
    (320.5, 143.4),
    (286.9, 128.7),
    (267.3, 121.3),
    (250.9, 121.3),
    (229.6, 127.8),
    (195.2, 145.9),
    (173.9, 161.4),
    (146.9, 172.1),
    (108.4, 181.9),
    (60.0, 186.0),
    (4.3, 189.3)
    ])
    tck, _ = splprep(ring_coords.T, s=0)
    u_fine = np.linspace(0, 1, 400)
    x_spline, y_spline = splev(u_fine, tck)

    spline_points = np.array(list(zip(x_spline, y_spline)))
    tip_point = np.array([rod_tip_x, rod_tip_y])
    distances = np.linalg.norm(spline_points - tip_point, axis=1)
    closest_index = np.argmin(distances)
    closest_spline_x, closest_spline_y = spline_points[closest_index]
    scale_pixels_per_mm = 2.04

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

    window_size = 5  # Should be odd
    half_window = window_size // 2
    start = max(0, new_index - half_window)
    end = min(len(spline_coords), new_index + half_window + 1)

    window_points = spline_coords[start:end]
    if len(window_points) < 2:
        tangent_new = np.array([1.0, 0.0])  # Fallback
    else:
        # Fit line to local spline segment using SVD
        mean_local = np.mean(window_points, axis=0)
        centered_local = window_points - mean_local
        _, _, vh_local = np.linalg.svd(centered_local, full_matrices=False)
        tangent_new = vh_local[0]

        # Optional: ensure it points forward (same x-direction as catheter_dir)
        if np.dot(tangent_new, catheter_dir) < 0:
            tangent_new *= -1


    unit_tangent_new = tangent_new / np.linalg.norm(tangent_new)
    alignment_angle = compute_angle(catheter_dir, unit_tangent_new)
    alignment_angle = min(alignment_angle, 180 - alignment_angle)
    catheter_to_x_angle = compute_angle(catheter_dir, np.array([1.0, 0.0]))
    print(f"alignment to the x-axis {catheter_to_x_angle}")
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
        # cv2.putText(image_rgb, "20mm along spline", (int(target_point[0]) + 10, int(target_point[1])),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        spline_poly = np.array(list(zip(x_spline, y_spline)), dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image_rgb, [spline_poly], isClosed=False, color=(0, 255, 0), thickness=1)
        # cv2.circle(image_rgb, rod_tip, 10, (255, 0, 0), -1)
        # cv2.putText(image_rgb, "Rod Tip (darkest)", (rod_tip_x + 10, rod_tip_y),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        # cv2.circle(image_rgb, (int(closest_spline_x), int(closest_spline_y)), 8, (0, 255, 255), -1)
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


    return tip, rod_tip, signed_distance_mm, desired_point, alignment_angle, catheter_to_x_angle


if __name__ == "__main__":
    image_path = new_capture()
    relation, tip_pos, dist_mm, spline_pt, angle, x_alignment = detect_rod_tip_yellow_right(image_path, graph=True)
    print("Tip position:", tip_pos)
    print("Distance (mm):", dist_mm)
    print("Angle (deg):", angle)
    print(tip_pos[0])
