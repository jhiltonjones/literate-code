import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def calculate_bending_angle(realimage, plot):
    image = cv2.imread(realimage, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image file '{realimage}' not found or cannot be opened.")
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    binary_mask_cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    binary_mask_cleaned = cv2.morphologyEx(binary_mask_cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)


    contours, _ = cv2.findContours(binary_mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if len(contours) > 0:
 
        largest_contour = contours[0].squeeze()
        x_vals = largest_contour[:, 0]
        y_vals = largest_contour[:, 1]

        valid_indices = (x_vals <= 900) & (x_vals > 0)
        x_vals = x_vals[valid_indices]
        y_vals = y_vals[valid_indices]

        sort_idx = np.argsort(x_vals)[::-1]
        x_vals = x_vals[sort_idx]
        y_vals = y_vals[sort_idx]

 
        def polynomial_loss(coeffs, x, y):
            poly = np.poly1d(coeffs)
            y_fit = poly(x)
            return np.sum((y - y_fit) ** 2)  

        def polynomial_loss_gradient(coeffs, x, y):
            poly = np.poly1d(coeffs)
            y_fit = poly(x)
            
            grad = np.zeros_like(coeffs)
            for i in range(len(coeffs)):  
                grad[i] = -2 * np.sum((y - y_fit) * (x ** (len(coeffs) - 1 - i)))
            
            return grad

        initial_coeffs = np.polyfit(x_vals, y_vals, 5)

        result = minimize(polynomial_loss, initial_coeffs, args=(x_vals, y_vals), method='BFGS', jac=polynomial_loss_gradient)

        optimized_coeffs = result.x

        poly_fit = np.poly1d(optimized_coeffs)
        x_fit_range = np.linspace(x_vals.min(), x_vals.max(), num=100)

        image_center_y = (image.shape[0] // 2 ) +30

        x_start = x_vals[0]
        x_mid_start = x_vals[len(x_vals) // 4]  
        x_mid_end = x_vals[(3 * len(x_vals)) // 4]  
        x_end = x_vals[-1]

        y_start, y_mid_start, y_mid_end, y_end = poly_fit([x_start, x_mid_start, x_mid_end, x_end])

        def compute_slope(x):
            return (5 * optimized_coeffs[0] * x**4 + 
                    4 * optimized_coeffs[1] * x**3 + 
                    3 * optimized_coeffs[2] * x**2 + 
                    2 * optimized_coeffs[3] * x + 
                    optimized_coeffs[4])
        
        slope_start = compute_slope(x_start)
        slope_mid_start = compute_slope(x_mid_start)
        slope_mid_end = compute_slope(x_mid_end)
        slope_end = compute_slope(x_end)

        theta_start = np.degrees(np.arctan(slope_start))  
        theta_mid_start = np.degrees(np.arctan(slope_mid_start))  
        theta_mid_end = np.degrees(np.arctan(slope_mid_end))  
        theta_end = np.degrees(np.arctan(slope_end))  
        
        theta_start_2 = np.degrees(np.arctan(slope_start))  
        theta_mid_start_2 = np.degrees(np.arctan(slope_mid_start))  
        theta_mid_end_2 = np.degrees(np.arctan(slope_mid_end))  
        theta_end_2 = np.degrees(np.arctan(slope_end))  

        bending_angle_deg_2 = max(abs(theta_start_2), abs(theta_mid_start_2), abs(theta_mid_end_2), abs(theta_end_2))

        bending_angle_deg = abs((theta_start + theta_end) / 2)
        if plot == True:
            plt.figure(figsize=(8, 6))
            plt.imshow(binary_mask_cleaned, cmap='gray')

            plt.plot(x_fit_range, poly_fit(x_fit_range), color='red', linewidth=2, label="Optimised Polynomial Curve")

            plt.axhline(y=image_center_y, color='green', linestyle='--', linewidth=2, label="Horizontal Center Line")

            plt.scatter([x_start, x_mid_start, x_mid_end, x_end], 
                        [y_start, y_mid_start, y_mid_end, y_end], 
                        color='yellow', s=100, label="Evaluation Points")

            plt.legend()
            plt.title(f"Bending Angle: {bending_angle_deg_2:.2f} degrees")
            
            plt.show(block=False)
            plt.pause(0.5)
            plt.close()
            # plt.show()

        # return bending_angle_deg, bending_angle_deg_2
        return bending_angle_deg_2


if __name__ == "__main__":
    realimage = "captured_image.jpg"
    plot = True
    bending_angle_deg_2 = calculate_bending_angle(realimage, plot)
    # print(f"Computed Bending Angle: {bending_angle_deg:.2f} degrees")
    print(f"Computed Bending Angle: {bending_angle_deg_2:.2f} degrees")
