"""
Script converts bag files into images.
"""
from argparse import ArgumentParser
from pathlib import Path

import pyrealsense2 as rs
import numpy as np
import cv2 as cv
# import os


parser = ArgumentParser("Script for converting realsense bag files into png and raw couples.", "python bags_to_images <path to dir with bag files>")
parser.add_argument("input", type=str, help="The bag file to convert.")
parser.add_argument("-o", type=str, help="Path of the output.", default="output", dest="output")
args = parser.parse_args()

# os.chdir(os.path.dirname(__file__))
# class args:
#     input = "../data/calib.bag"
#     output = "./output/"


file = args.input
output = Path(args.output)
output.mkdir(exist_ok=True, parents=True)

print(f"Processing {file}")
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(file, repeat_playback=False)
pipeline.start(config)

output_color = output.joinpath("png")
output_depth = output.joinpath("depth")
output_color.mkdir(exist_ok=True)
output_depth.mkdir(exist_ok=True)

processed = set()

while True:
    try:
        frames = pipeline.wait_for_frames()
    except RuntimeError:  # No frames left
        break

    depth_frame = frames.get_depth_frame()

    frame_number = depth_frame.get_frame_number()
    frame_name = f"frame_{frame_number:04d}.png"
    if frame_number in processed:
        continue
    
    processed.add(frame_number)
    rgb_frame = frames.get_color_frame()
    depth_frame = np.asanyarray(depth_frame.get_data())
    rgb_frame = np.asanyarray(rgb_frame.get_data())
    rgb_frame = rgb_frame[:, :, ::-1]

    cv.imwrite(str(output_color.joinpath(frame_name)), rgb_frame)
    cv.imwrite(str(output_depth.joinpath(frame_name)), depth_frame)
