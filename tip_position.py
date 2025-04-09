import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image_path = "/home/jack/literate-code/captured_image1.jpg"
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold to isolate dark objects (invert: rod becomes white)
_, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Assume largest contour is the rod
rod_contour = max(contours, key=cv2.contourArea)

# Create a white canvas
segmented = np.ones_like(gray) * 255

# Draw the rod as black on white
cv2.drawContours(segmented, [rod_contour], -1, color=0, thickness=-1)

# Discard all x > 1200 by setting them to white
segmented[:, 1250:] = 255

# Show result
plt.figure(figsize=(10, 6))
plt.imshow(segmented, cmap='gray')
plt.title('Segmented Rod (Rod is Black, x > 1200 Removed)')
plt.axis('off')
plt.show()
