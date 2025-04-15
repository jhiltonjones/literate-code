import cv2
import matplotlib.pyplot as plt

# Load image
image_path = 'captured_image1.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Store clicked points
clicked_points = []

def onclick(event):
    if event.xdata and event.ydata:
        clicked_points.append((event.xdata, event.ydata))
        print(f"Point {len(clicked_points)}: ({event.xdata:.2f}, {event.ydata:.2f})")

        if len(clicked_points) == 2:
            plt.close()

# Show image and collect points
fig, ax = plt.subplots()
ax.imshow(image_rgb)
ax.set_title("Click two points to define a distance")
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

# If two points clicked
if len(clicked_points) == 2:
    import numpy as np

    # Euclidean distance in pixels
    p1, p2 = clicked_points
    pixel_distance = np.linalg.norm(np.array(p1) - np.array(p2))
    print(f"\nPixel distance between points: {pixel_distance:.2f} pixels")

    # Ask user for real-world distance
    real_distance = float(input("Enter the real-world distance between the points (e.g., in mm): "))
    scale = pixel_distance / real_distance
    print(f"\nScale: {scale:.2f} pixels per real-world unit")
