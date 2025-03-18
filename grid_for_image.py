import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load the newly uploaded image
image_path = "captured_image.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# Apply adaptive thresholding to segment the rod (darkest region)
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

# Create a mask where the rod is detected as black and the background is white
mask = np.ones_like(img) * 255  # Initialize a white mask
mask[thresh == 255] = 0  # Keep the black rod as it is

# Display the masked image
plt.figure(figsize=(10, 10))
plt.imshow(mask, cmap='gray')
plt.title("Segmented Black Rod Mask")
plt.axis("off")
plt.show()

# Save the mask
masked_image_path = "/mnt/data/rod_mask.jpg"
cv2.imwrite(masked_image_path, mask)

masked_image_path
