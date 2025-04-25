import cv2

def new_capture(filename='focused_image.jpg', focus=255):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, focus)

    ret, frame = cap.read()
    cap.release()

    if ret:
        cv2.imwrite(filename, frame)
        print(f"Focused image captured and saved as {filename}")
        return filename
    else:
        raise RuntimeError("Failed to capture image")
if __name__ == "__main__":
    new_capture()