from __future__ import annotations  # Disable automatic annotation evaluation. For compatibility with Python < 3.10.
from torchvision.transforms import Resize
from collections import defaultdict

from .calibrator import *
from .depthcamera import *
from .data import *

import ultralytics as ul
import numpy as np
import cv2 as cv
import torch


__all__ = ("Camera",)


class Camera:
    """
    Main camera object.

    Parameters
    -------------
    intrinsics: Intrinsics
        The intrinsic camera parameters.
    affine_align: np.matrix[2, 3]
        Affine matrix for aligning depth points to rgb points.
        This should be a 2x3 matrix (without the final [0, 0, 1] row).
        Defaults to np.eye(2, 3) (alignment has no effect).
    H_cam_robot: np.matrix[4, 4]
        Transformation matrix which transforms a point in the camera coordinate system into the robot's coordinate
        system through equation H_cam_robot x P_camera.
        Defaults to np.identity(4) (transformation has no effect).
    detect_model_path: str
        Path to the YOLO model responsible for detecting elevator state and transport gate state.
        Defaults to yolov8n-seg.pt (auto-downloaded).
    obstacle_model_path: str
        Path to the FastSAM model responsible for segmenting objects. It is used for obstacle detection.
        Defaults to FastSAM-s.pt (auto-downloaded).
    device: str | torch.device
        PyTorch argument instructing it the device to use.
        This can be e.g., 'cpu', 'mps', 'cuda' or e.g. 'cuda:0' for the cuda device with index 0.
        It can also be a torch.device object. 
        Defaults to 'cuda'.
    """
    def __init__(
        self,
        camera_matrix: np.ndarray,
        affine_align: np.ndarray = np.identity(3),
        H_cam_robot: np.ndarray = np.identity(4),
        detect_model_path: str = "yolov8n-seg.pt",
        obstacle_model_path: str = "FastSAM-s.pt",
        device: str | torch.device = "cuda"
    ) -> None:
        self.detect_model = ul.YOLO(detect_model_path)
        self.obstacle_model = ul.FastSAM(obstacle_model_path)
        self.device = device

        self.H_cam_robot = H_cam_robot
        self.depth = DepthCamera(5, affine_align)
        self.calib = Calibrator()

        self.camera_matrix = camera_matrix
        self.inv_camera_matrix = np.linalg.inv(camera_matrix)

        # Dictionary for keeping track of how many frames each object has been detected
        self.detection_count: dict[int, int] = defaultdict(lambda: 0)  # {id: count}

    def pixel_to_xyz(self, xi: np.ndarray, yi: np.ndarray, depth: int) -> np.ndarray:
        """
        Transforms a pixel and its corresponding depth into XYZ in the camera coordinate system.
        The camera coordinate system is as follows (if looking in the same direction as the camera is capturing):

        - x: pointing right
        - y: pointing down
        - z: pointing forward

        Parameters
        -----------
        xi: int
            Horizontal pixel location.
        yi: int
            Vertical pixel location.
        depth: int
            Depth (in millimeters) corresponding to the (xi, yi) pixel.
        """
        if not isinstance(xi, np.ndarray):  # Arguments are a single coordinate, not a list of them
            xi = np.array([xi])
            yi = np.array([yi])

        xyz: np.ndarray = (self.inv_camera_matrix @ np.vstack([xi, yi, np.ones(len(xi))]))
        xyz[0, :] *= depth
        xyz[1, :] *= depth
        xyz[2, :] *= depth

        return xyz

    def transform_to_robot_xyz(self, x: np.ndarray | int, y: np.ndarray | int, z: np.ndarray | int):
        """
        Transforms camera XYZ into robot's xyz coordinate system.
        """
        if not isinstance(x, np.ndarray):  # Arguments are a single coordinate, not a list of them
            x = np.array([x])
            y = np.array([y])
            z = np.array([z])

        stacked = np.vstack((x, y, z, np.ones(len(x))))
        ret_: np.ndarray = (self.H_cam_robot @ stacked)[:3]
        if len(ret_[0]) == 1:  # Single coordinate
            ret_ = ret_.flatten()

        return ret_

    def detect_obstacles(
        self,
        img: np.ndarray,
        img_depth: np.ndarray,
        max_distance_mm: float = 3000,
        n_filter_frames: int = 10,
        align: bool = False,
        draw: bool = False
    ) -> list[ObstacleResult]:
        """
        Detects potential objects in the way.

        Parameters
        ------------
        img: np.ndarray
            Normal camera frame (rgb / grayscale).
        img_depth: np.ndarray
            Depth frame containing information about the depth at each pixel.
        max_distance_mm: float
            The maximum distance in mm for the object to be considered an obstacle.
            Objects outside this range will be ignored.
            Defaults to 3000 mm.
        n_filter_frames: int
            Number of frames individual objects need to be detected by the model before
            considering them as valid detections.
            When the object is detected less than these number of frames, it is ignored.
            Defaults to 10.
        align: bool
            Whether to align the ``img_depth`` to ``img``.
            Defaults to ``False`` as it is assumed the image is already aligned.
        draw: bool
            Whether to draw detected objects. When ``True``,
            the returned result will have a ``frame`` attribute set to the drawn frame.
            Defaults to False.

            WARNING: This reduces FPS. Its is recommended to be for testing only.
        """
        frames = []
        obstacles = []

        depth = self.depth

        depth.sample(img_depth)
        img_depth = depth.fill_holes(img_depth)

        if align:
            img_depth = depth.align(img_depth)

        if draw:
            frames = [
                rgb_draw_frame := img.copy(),
                segment_frame := np.zeros((*img.shape[:2], 3), dtype=np.uint8),
                depth_draw_frame := cv.cvtColor(
                    depth.normalize(img_depth[:img.shape[0], :img.shape[1]], histogram=False),
                    cv.COLOR_GRAY2BGR
                )
            ]

        result = self.obstacle_model.track(img, persist=True, device=self.device, verbose=False)[0]
        if result.masks.shape[1:] != img.shape[:2]:
            result.masks.data = Resize(img.shape[:2])(result.masks.data)


        result.masks.data = result.masks.data.type(torch.bool)
        result = result.cpu().numpy()
        boxes = result.boxes
        masks = result.masks or []  # If there are no detections, masks will be None, thus the OR operator returns []

        detected_ids = set()  # Track detected ids in order to remove all the keys from the detection counts, that weren't detected
        for i, (box, mask) in enumerate(zip(boxes, masks)):
            cx, cy, width, height = box.xywh[0]
            center = np.intp((cx, cy))

            mask_data: np.ndarray = mask.data[0]

            # Calculate n coordinates and take mean of the 10% of closest ones
            decimation = 10
            yi, xi = mask_data[::decimation, ::decimation].nonzero()
            yi *= decimation
            xi *= decimation

            depth_points = img_depth[yi, xi]

            # Keep only the points whose depth is non-zero
            zero_mask = (depth_points != 0)
            depth_points = depth_points[zero_mask]
            xi, yi = xi[zero_mask], yi[zero_mask]

            if len(xi) < 2:
                continue

            # Filter-out points that were leaked to occluded objects due to bad
            # segmentation mask
            indices = np.argsort(depth_points)
            gr_sort = np.abs(np.gradient(depth_points[indices]))
            spikes = gr_sort > np.median(gr_sort) * 3
            ## Create different areas based on the gradient of depth data.
            areas = [area := []]
            last_s = spikes[0]
            for s, i in zip(spikes, indices):
                if s:
                    if last_s:
                        continue

                    areas.append(area := [])

                last_s = s
                area.append(i)               

            ## Take the area that is the greatest.
            area_max = max(areas, key=lambda k: len(k))

            depth_points = depth_points[area_max]
            xi, yi = xi[area_max], yi[area_max]

            # Take the top 10% points and also points that deviate max 10% above the 10th percentile.
            p10 = np.percentile(depth_points, 10, method="nearest")
            km = (depth_points <= p10 * 1.10)
            if not km.shape:
                continue

            xc, yc, zc = self.pixel_to_xyz(xi, yi, depth_points)

            # Final location is defined as median of the coordinates with z below 10th percentile
            xcf, ycf, zcf = np.median([xc, yc, zc], axis=1)

            # Limit detections to objects close enough to the camera, otherwise the object isn't yet an obstacle.
            if zcf > max_distance_mm:
                continue

            robot_xyz = self.transform_to_robot_xyz(xcf, ycf, zcf)

            # Prevent detection of the floor.
            center_xyz = self.pixel_to_xyz(*center, img_depth[center[1], center[0]])
            center_robot_xyz = self.transform_to_robot_xyz(*center_xyz).flatten()
            if center_robot_xyz[2] <= 5:
                off = center.copy()
                off[1] = np.intp(off[1] + height * 0.2)
                off = np.clip(off, [0, 0], np.intp(img_depth.shape[::-1]) - 1)

                off_camera_xyz = self.pixel_to_xyz(*off, img_depth[off[1], off[0]])
                off_robot_xyz = self.transform_to_robot_xyz(*off_camera_xyz).flatten()

                # Z-axis of original center point will always be higher than the offset point, unless the points
                # are both on the floor and some noise is present (without noise it should be 0).
                # If offset is greater than 300, that indicates some depth image quality issue. Only small objects
                # would have their center at z < 5mm, thus the points should be really close together. If we get
                # a big difference, that means there are probably some reflections that are causing issues.
                offset = center_robot_xyz[2] - off_robot_xyz[2]
                if offset < 10 or offset > 300:
                    continue

            if box.id is not None:
                track_id = int(box.id[0])
            else:
                track_id = -1

            obj_width = obj_height = 0

            # All the non filtered conditions are met, now update frame filtering counts
            detected_ids.add(track_id)
            self.detection_count[track_id] += 1
            if self.detection_count[track_id] < n_filter_frames:
                continue

            robot_xyz = robot_xyz.astype(np.int32)
            obstacles.append(ObstacleResult(robot_xyz, obj_width, obj_height, track_id))

            if draw:
                color = [64 + track_id * 2 % 192, 64 + track_id * 4 % 192, 128 + track_id * 8 % 192]
                segment_frame[mask_data] = color

                for xx, yy in zip(xi, yi):
                    cv.circle(rgb_draw_frame, (xx, yy), 1, color)

                for i, c in enumerate(('x', 'y', 'z')):
                    cv.putText(
                        rgb_draw_frame,
                        f"{c}: {robot_xyz[i]}",
                        center + np.array([-25, i*20 + 20], dtype=np.intp),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (128, 128, 255),
                        1
                    )

                cv.putText(
                    rgb_draw_frame,
                    f"id={track_id}",
                    center + np.array([0, -20], dtype=np.intp),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (128, 0, 128),
                    1
                )
                cv.circle(rgb_draw_frame, center, 5, (0, 0, 255))
                cv.circle(depth_draw_frame, center, 5, (0, 0, 255))
                cv.circle(segment_frame, center, 5, (0, 0, 255))
                cv.putText(
                    segment_frame,
                    f"h={obj_height} mm",
                    center + np.array([15, -20], dtype=np.intp),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (128, 64, 0),
                    1
                )
                cv.putText(
                    segment_frame,
                    f"w={obj_width} mm",
                    center + np.array([15, -35], dtype=np.intp),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 128, 128),
                    1
                )

        # Filter out objects detected in current frame and reset counts (by deleting the key) of non-detected ones
        non_detected_ids = set(self.detection_count.keys()) - detected_ids
        for id in non_detected_ids:
            del self.detection_count[id]

        return obstacles, frames

    def detect_yolo(self, img: np.ndarray, draw: bool = False) -> list[DoorResult]:
        """
        Detects whether something is opened / closed.
        """
        result = self.detect_model(img, verbose=False, conf=0.75)[0]
        boxes = result.boxes

        # Detectable image classes are created by inheriting DoorResult 
        # Get a list of all DoorResult inherited classes
        valid_classes = DoorResult.__subclasses__()

        if draw:
            rgb_draw_frame = result.plot()
            frames = [rgb_draw_frame]
        else:
            frames = []

        detections = []

        boxes = boxes.cpu()
        for cls in valid_classes:
            opened_id, closed_id = cls.ids
            for box in boxes:
                class_id = int(box.cls[0].item())
                if class_id == opened_id:
                    detections.append(cls(True))
                elif class_id == closed_id:
                    detections.append(cls(False))

        return detections, frames
