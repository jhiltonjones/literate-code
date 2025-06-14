# import cv2

# def new_capture(filename='focused_image.jpg', focus=255):
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         raise RuntimeError("Cannot open camera")

#     cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
#     cap.set(cv2.CAP_PROP_FOCUS, focus)

#     ret, frame = cap.read()
#     cap.release()

#     if ret:
#         cv2.imwrite(filename, frame)
#         print(f"Focused image captured and saved as {filename}")
#         return filename
#     else:
#         raise RuntimeError("Failed to capture image")
# if __name__ == "__main__":
#     new_capture()

import cv2

def new_capture(filename='focused_image.jpg', focus=255):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    # Try disabling autofocus (not all cameras support this)
    if cap.set(cv2.CAP_PROP_AUTOFOCUS, 0):
        print("Autofocus disabled.")
    else:
        print("Warning: Could not disable autofocus (unsupported).")

    # Try setting manual focus (value typically from 0 to 255)
    if cap.set(cv2.CAP_PROP_FOCUS, float(focus)):
        print(f"Focus set to {focus}.")
    else:
        print("Warning: Could not set focus manually (unsupported).")

    # Optional: Disable auto exposure and auto white balance for consistent lighting
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual mode on many webcams
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)         # May need tuning per device

    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)

    # Wait briefly to let camera apply new settings
    for _ in range(5):
        cap.read()

    # Capture final frame
    ret, frame = cap.read()
    cap.release()

    if ret and frame is not None:
        cv2.imwrite(filename, frame)
        print(f"Focused image captured and saved as {filename}")
        return filename
    else:
        raise RuntimeError("Failed to capture image")

if __name__ == "__main__":
    new_capture()
