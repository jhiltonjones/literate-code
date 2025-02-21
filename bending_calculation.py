
import cv2
import numpy as np
import matplotlib.pyplot as plt


def calculate_bending_angle(realimage):
    image_path = realimage
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    _, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    binary_mask_cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    binary_mask_cleaned = cv2.morphologyEx(binary_mask_cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(binary_mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    from scipy.interpolate import CubicSpline


    kernel_large = np.ones((5, 5), np.uint8)  
    binary_mask_cleaned = cv2.morphologyEx(binary_mask_cleaned, cv2.MORPH_OPEN, kernel_large, iterations=3)

    contours, _ = cv2.findContours(binary_mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)  

    if len(contours) > 0:
        largest_contour = contours[0].squeeze()
        x_vals = largest_contour[:, 0]
        y_vals = largest_contour[:, 1]

        sort_idx = np.argsort(x_vals)
        x_vals = x_vals[sort_idx]
        y_vals = y_vals[sort_idx]

    poly_coeffs = np.polyfit(x_vals, y_vals, 3)  
    cubic_fit = np.poly1d(poly_coeffs)

    x_fit_range = np.linspace(x_vals.min(), x_vals.max(), num=100)

    key_indices = np.linspace(0, len(x_vals) - 1, num=5, dtype=int)
    key_x = x_vals[key_indices]
    key_y = cubic_fit(key_x)

    slopes = [3 * poly_coeffs[0] * x**2 + 2 * poly_coeffs[1] * x + poly_coeffs[2] for x in key_x]

    angles = [np.arctan(abs((slopes[i+1] - slopes[i]) / (1 + slopes[i] * slopes[i+1]))) for i in range(len(slopes) - 1)]

    bending_angle_deg = np.degrees(np.mean(angles))

    # plt.figure(figsize=(8, 6))
    # plt.imshow(binary_mask_cleaned, cmap='gray')
    # plt.plot(x_fit_range, cubic_fit(x_fit_range), color='red', linewidth=2, label="Fitted Cubic Curve")  # Limited range
    # plt.scatter(key_x, key_y, color='blue', label="Key Points")
    # plt.legend()
    # plt.title(f"Corrected Bending Angle: {bending_angle_deg:.2f} degrees")
    # plt.show()
    return bending_angle_deg

if __name__ == "__main__":
    realimage = "captured_image.jpg"
    actual_angle = -1 * calculate_bending_angle(realimage)
    print(actual_angle)




