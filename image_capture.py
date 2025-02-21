from pypylon import pylon
import cv2
import numpy as np
import os

def capture_image():
    # Create a camera instance
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

    # Open the camera
    camera.Open()

    try:
        # Start grabbing
        camera.StartGrabbing()
        
        # Retrieve the image
        grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grab_result.GrabSucceeded():
            # Convert image buffer to NumPy array
            img = grab_result.Array

            # Convert grayscale images to BGR format
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # Define the image save path
            image_filename = "captured_image.jpg"
            image_path = os.path.abspath(image_filename)

            # Save the image
            cv2.imwrite(image_path, img)
            print(f"Image saved at: {image_path}")

            return image_path  # Return the full path of the saved image

    finally:
        # Ensure resources are released
        grab_result.Release()
        camera.StopGrabbing()
        camera.Close()

    print("Camera released. Exiting script.")
    return None  # Return None if the image wasn't captured successfully

if __name__== "__main__":
    image_path = capture_image()
    print(f"Captured image is stored at: {image_path}")



