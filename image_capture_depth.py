import pyrealsense2 as rs
import numpy as np
import cv2

# Configure the pipeline for RealSense camera
pipeline = rs.pipeline()
config = rs.config()

# Enable the RGB stream (color camera)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    # Wait for a frame
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    
    if not color_frame:
        print("No color frame captured")
        exit()

    # Convert to NumPy array
    color_image = np.asanyarray(color_frame.get_data())

    # Save the image
    cv2.imwrite("realsense_2d_image.png", color_image)

    # Display the image
    cv2.imshow("Captured Image", color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

finally:
    pipeline.stop()
