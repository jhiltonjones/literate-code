from pypylon import pylon
import cv2
import numpy as np
import os

def capture_image():
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

    camera.Open()

    try:
        camera.StartGrabbing()
        
        grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grab_result.GrabSucceeded():
            img = grab_result.Array

            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            image_filename = "captured_image.jpg"
            image_path = os.path.abspath(image_filename)

            cv2.imwrite(image_path, img)
            print(f"Image saved at: {image_path}")

            return image_path  

    finally:
        grab_result.Release()
        camera.StopGrabbing()
        camera.Close()

    print("Camera released. Exiting script.")
    return None  

if __name__== "__main__":
    image_path = capture_image()
    print(f"Captured image is stored at: {image_path}")



