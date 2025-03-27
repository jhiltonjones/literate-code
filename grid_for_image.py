import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splprep, splev

# Step 1: Load image
image_path = 'captured_image.jpg'  # <- Replace with your path if needed
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 2: Interactive point selection
plt.figure(figsize=(10, 6))
plt.imshow(image_rgb)
plt.title("Click on the center of each ring (in order). Close the window when done.")
ring_coords = plt.ginput(n=-1, timeout=0)  # You can right-click or press enter to finish
plt.close()

ring_coords = np.array(ring_coords)
x, y = ring_coords[:, 0], ring_coords[:, 1]

# Step 3: Fit a spline through the points
tck, _ = splprep([x, y], s=0)
u_fine = np.linspace(0, 1, 400)
x_fine, y_fine = splev(u_fine, tck)

# Step 4: Plot result
plt.figure(figsize=(10, 6))
plt.imshow(image_rgb)
plt.plot(x, y, 'ro', label='Selected Ring Centers')
plt.plot(x_fine, y_fine, 'b-', label='Fitted Spline')
plt.legend()
plt.title("Spline Fit Through Rings")
plt.axis('off')
plt.show()

# Optional: print coordinates
print("Selected ring coordinates (x, y):")
for i, (xi, yi) in enumerate(zip(x, y)):
    print(f"Ring {i+1}: ({xi:.1f}, {yi:.1f})")
