import cv2
import os

OUT_DIR = "/home/jack/literate-code/calib_images"
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    cap = cv2.VideoCapture(1)  # change index if you have multiple cameras

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press 'c' to capture an image, 'q' to quit.")
    img_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow("Calibration capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            filename = os.path.join(OUT_DIR, f"calib_{img_id:03d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
            img_id += 1

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
