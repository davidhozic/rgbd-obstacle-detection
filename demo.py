from __future__ import annotations
from argparse import ArgumentParser
from enum import IntEnum, auto
from collections import deque
from pathlib import Path

import numpy as np
import cv2 as cv
import warnings
import time
import os


class ModelTask(IntEnum):
    DETECT_OBSTACLES = 0
    DETECT_DOORS = auto()
    CALIB_ALIGN = auto()
    CALIB_CAMERA = auto()


class ImageWriter:
    def __init__(self, path: str | Path, *args):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.cnt = 0

    def write(self, img: np.ndarray):
        if self.cnt % 25 == 0:
            output = self.path.joinpath(f"frame_{self.cnt}.png")
            cv.imwrite(str(output), img)

        self.cnt += 1


def main(input_path: str, task: ModelTask, output_path: str | Path = None):
    # Import here to speed up the time it takes for the argument parser
    # to display its information. They are needed only if everything was entered correctly.
    from camera import Camera
    import numpy as np
    import cv2 as cv
    import torch


    # Affine matrix from depth image to RGB. Used for aligning the depth image with RGB.
    H_affine = np.array([
        [ 1.41369034e+00,  2.61865419e-02, -2.76181195e+02],
        [ 4.39939439e-03,  1.41815829e+00, -8.79810702e+01],
        [ 0,               0,               1,            ],
    ])

    # Transformation matrix from camera's system into robot's system.
    angle_x = np.deg2rad(0)
    camera_height = 360

    # For testna_3 video
    # angle_x = np.deg2rad(-3)
    # camera_height = 345

    ct = np.cos(angle_x)
    st = np.sin(angle_x)
    R_cam_x = np.array([
        [1,   0,   0],
        [0,  ct, -st],
        [0,  st,  ct]
    ])

    R_cam_robot = np.array([
        [ 0,  0,  1],
        [-1,  0,  0],
        [ 0, -1,  0]
    ])

    H_cam_robot = np.zeros((4, 4))
    H_cam_robot[:3, :3] = R_cam_robot @ R_cam_x
    H_cam_robot[:, 3] = [0, 0, camera_height, 1]  # Translational part
    # H_cam_robot = np.matrix([
    #     [ 0,  0,  1,    0],
    #     [-1,  0,  0,    0],
    #     [ 0, -1,  0,  360],
    #     [ 0, 0,   0,    1],
    # ])

    camera_matrix = np.array([
        [610.3503, 0,           324.2132],
        [0,        614.0150,    250.7816],
        [0,        0,           1       ]
    ])

    if torch.cuda.is_available():
        device = "cuda"
    else:
        warnings.warn("CUDA is not available.")
        device = "cpu"

    camera = Camera(
        camera_matrix,
        H_affine, H_cam_robot,
        "models/vrata.pt", "models/FastSAM-s.pt",
        device=device
    )

    running_fps = deque([0] * 5, maxlen=5)
    rgb_path = os.path.join(input_path, "png")
    if not os.path.exists(rgb_path):
        rgb_path = os.path.join(input_path, "rgb")

    depth_path = os.path.join(input_path, "depth")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True, parents=True)

    if task is ModelTask.CALIB_CAMERA:
        if output_path is not None:
            output_path = output_path.joinpath("calibration")

        print(camera.calib.rgb_intrinsics(input_path, debug_path=output_path))
        return

    writer: None | cv.VideoWriter = None
    for file in sorted(os.listdir(rgb_path)):
        file_depth = os.path.join(depth_path, file)
        file = os.path.join(rgb_path, file)

        if not (os.path.exists(file_depth) and os.path.exists(file)):
            break

        frame = cv.imread(file)
        frame_depth = cv.imread(file_depth, cv.IMREAD_UNCHANGED)

        # 8 bit depth images need to be scaled back to 16 bit.
        # (This still doesn't produce the same results as native 16 bit images).
        n_bits = frame_depth[0, 0].nbytes * 8
        if n_bits < 16:
            frame_depth = frame_depth / (2**(n_bits) - 1) * (2**16 - 1)

        start = time.perf_counter()
        if task is ModelTask.DETECT_OBSTACLES:
            result, frames = camera.detect_obstacles(frame, frame_depth, align=True, draw=True, max_distance_mm=6500)
        elif task is ModelTask.DETECT_DOORS:
            result, frames = camera.detect_yolo(frame, draw=True)
        elif task is ModelTask.CALIB_ALIGN:
            print(camera.calib.depth_alignment(frame, frame_depth))
            break

        end = time.perf_counter()

        if frames:
            diff = end - start
            fps = 1 / diff
            running_fps.append(fps)
            fps = int(np.mean(running_fps))

            frame = frames[0]
            cv.putText(
                frame,
                f"FPS: {fps}",
                np.array([10, int(frame.shape[0] * 0.98)]),
                cv.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 0, 0), 2
            )

        frames_o = []
        for i, frame in enumerate(frames):
            cv.imshow(f"Frame {i}", frame)
            if frame.shape[-1] != 3:
                frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)

            frames_o.append(frame)

        frames = frames_o
        if output_path:
            resulting_frame = np.hstack(frames)
            aspect_ratio = resulting_frame.shape[1] / resulting_frame.shape[0]
            resulting_frame = cv.resize(resulting_frame, (1920, int(1920 / aspect_ratio)))

            if writer is None:
                filename = Path(os.path.basename(input_path.rstrip('\\/')))
                writer = cv.VideoWriter(
                    os.path.join(output_path, f"{filename}-[{task.name.lower()}].mp4"),
                    cv.VideoWriter_fourcc(*"avc1"),
                    24,
                    resulting_frame.shape[1::-1],
                    True
                )

            writer.write(resulting_frame)

        if cv.waitKey(1) == ord('q'):
            break

    cv.destroyAllWindows()

if __name__ == "__main__":
    parser = ArgumentParser(
        "Demo script for Robot Vision Seminar. (David Hožič, Blaž Primšar)",
        "python demo.py --task detect_obstacles --input ./data/testni_podatki/testna_1/ --output=<optional-folder>"
    )

    valid_tasks = [x.name.lower() for x in ModelTask]
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--task", type=str, choices=valid_tasks, required=True)

    # class FakeArgs:
    #     input = r".\data\testni_podatki\testna_1"
    #     task = "detect_obstacles"
    #     output = None#"output"

    # args = FakeArgs()

    args = parser.parse_args()

    task_in = args.task.upper()
    task = None
    for e in ModelTask:
        if e.name == task_in:
            task = e
            break

    main(args.input, task, args.output)
