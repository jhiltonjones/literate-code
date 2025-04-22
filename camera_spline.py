import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splprep, splev
from new_cam import new_capture

def detect_rod_tip(image_path, graph=True):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhance contrast for clearer tip detection
    gray = cv2.equalizeHist(gray)

    # Adaptive thresholding for robust tip detection
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assume the rod tip is the smallest leftmost contour
    rod_tip = None
    min_x = np.inf
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 10 < area < 500:  # area filter to remove noise
            x, y, w, h = cv2.boundingRect(cnt)
            if x < min_x:
                min_x = x
                rod_tip = (x, y + h // 2)

    if rod_tip is None:
        raise ValueError("Rod tip not detected")

    rod_tip_x, rod_tip_y = rod_tip

    # --- Fit spline through given ring points ---
    ring_coords = np.array([
        (614.0, 229.1), (539.2, 225.0), (463.4, 217.7),
        (413.5, 201.1), (357.4, 172.0), (306.5, 146.0),
        (268.1, 148.1), (217.2, 174.0), (127.8, 242.6), (54.0, 302.9)
    ])
    tck, _ = splprep(ring_coords.T, s=0)
    u_fine = np.linspace(0, 1, 400)
    spline_coords = np.array(splev(u_fine, tck)).T

    # Closest spline point
    distances = np.linalg.norm(spline_coords - rod_tip, axis=1)
    closest_idx = np.argmin(distances)
    closest_spline_pt = spline_coords[closest_idx]

    # Signed distance
    pixel_distance = distances[closest_idx]
    scale_pixels_per_mm = 9.55
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
        plt.title('Rod Tip Detection')
        plt.show()

    print(f"Signed distance from rod tip to spline: {signed_distance_mm:.2f} mm")
    print(relation_text)

    return relation_text, rod_tip, signed_distance_mm, closest_spline_pt


if __name__ == "__main__":
    image_path = new_capture()
    relation, tip_pos, dist_mm, spline_pt = detect_rod_tip(image_path, graph=True)
    print("Tip position:", tip_pos)
    print("Distance (mm):", dist_mm)