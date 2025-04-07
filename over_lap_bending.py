from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
# Load all images
image_paths = [
    "/home/jack/literate-code/captured_image0.jpg",
    "/home/jack/literate-code/captured_image10.jpg",
    "/home/jack/literate-code/captured_image20.jpg",
    "/home/jack/literate-code/captured_image30.jpg",
    "/home/jack/literate-code/captured_image40.jpg",
    "/home/jack/literate-code/captured_image50.jpg",
    "/home/jack/literate-code/captured_image60.jpg",
    "/home/jack/literate-code/captured_image70.jpg",
    "/home/jack/literate-code/captured_image80.jpg",
    "/home/jack/literate-code/captured_image90.jpg"
]

# Convert images to grayscale arrays and stack them
images = [np.array(Image.open(path).convert("L"), dtype=np.float32) for path in image_paths]

# Average the images
average_image = np.mean(images, axis=0)

# Normalize to uint8
# Enhance the contrast by scaling the pixel values (making darker features stand out more)
# We'll multiply by a contrast factor < 1 to make the image appear darker overall
# contrast_factor = 1.5
# darker_image = average_image * contrast_factor

# # Clip and convert to uint8 for display
# darker_image_uint8 = np.clip(darker_image, 0, 255).astype(np.uint8)

# Define a threshold: only darken pixels that are already below this intensity in the region
# Copy the original average image
enhanced_dark_region = average_image.copy()

# Redefine the mask using the polygon for completeness
polygon = np.array([[1000, 442], [1179, 435], [1213, 167], [986, 183]])
mask = np.zeros_like(average_image, dtype=np.uint8)
cv2.fillPoly(mask, [polygon], 255)

# Redefine the contrast factor and threshold
contrast_factor = 0.8
dark_threshold = 90

# Copy the original image for enhancement
enhanced_dark_region = average_image.copy()

# Apply the mask and threshold
mask_bool = mask == 255
dark_pixels = (enhanced_dark_region < dark_threshold) & mask_bool

# Apply darkening only to dark pixels within the region
enhanced_dark_region[dark_pixels] *= contrast_factor

# Clip and convert to uint8
enhanced_dark_region_uint8 = np.clip(enhanced_dark_region, 0, 255).astype(np.uint8)

# Show the result
plt.imshow(enhanced_dark_region_uint8, cmap='gray')
plt.axis('off')
plt.title("Regionally Enhanced Dark Pixels (Fixed Mask)")
plt.show()



