import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splprep, splev
from new_cam import new_capture

image_path = new_capture()
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 6))
plt.imshow(image_rgb)
plt.title("Click on the center of each ring (in order). Close the window when done.")
ring_coords = plt.ginput(n=-1, timeout=0) 
plt.close()

ring_coords = np.array(ring_coords)
x, y = ring_coords[:, 0], ring_coords[:, 1]

tck, _ = splprep([x, y], s=0)
u_fine = np.linspace(0, 1, 400)
x_fine, y_fine = splev(u_fine, tck)

plt.figure(figsize=(10, 6))
plt.imshow(image_rgb)
plt.plot(x, y, 'ro', label='Selected Ring Centers')
plt.plot(x_fine, y_fine, 'b-', label='Fitted Spline')
plt.legend()
plt.title("Spline Fit Through Rings")
plt.axis('off')
plt.show()

print("Selected ring coordinates (x, y):")
for i, (xi, yi) in enumerate(zip(x, y)):
    print(f"({xi:.1f}, {yi:.1f}),")
