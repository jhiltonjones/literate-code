import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file("/home/jack/Documents/20250531_120627.bag")
pipeline.start(config)

try:
    for _ in range(100):  # read first 100 frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        print("Got frame:", color_frame.get_width(), "x", color_frame.get_height())
finally:
    pipeline.stop()
