from __future__ import annotations
from matplotlib.backend_bases import MouseEvent
from pathlib import Path

from .depthcamera import DepthCamera

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os


__all__ = ("Calibrator",)


class Calibrator:
    """
    Used for performing various camera calibrations.
    """
    @staticmethod
    def depth_alignment(img: np.ndarray, img_depth: np.ndarray):
        """
        Opens a window, where three points can be selected in order to perform
        alignment of depth and rgb modules.

        Returns
        ---------
        np.ndarray
            Affine transformation matrix. This matrix transforms depth module coordinates
            into rgb module coordinates.
        """
        img_depth = DepthCamera.normalize(img_depth)  # Improve the contrast

        def align(event: MouseEvent):
            nonlocal idx

            if event.dblclick:
                idx += 1
                if idx == len(images):
                    plt.close()
                    return

                ax.clear()
                prev = np.array(pts[idx-1])
                for i, xy in enumerate(prev, 1):
                    ax.text(*xy, i, color="white", bbox=dict(boxstyle="circle"))

                ax.imshow(images[idx], cmap="gray")

            elif event.key == "shift":
                if event.xdata is None:
                    return

                pt = event.xdata, event.ydata
                ax.scatter(*pt, c='r')
                pts[idx].append(pt)

            fig.canvas.draw()

        ax: plt.Axes
        fig: plt.Figure
        fig, ax = plt.subplots()
        pts = [[], []]
        images = [img, img_depth]
        idx = 0
        ax.imshow(images[idx])
        fig.suptitle("Shift: Create point\nDouble click: Next image")
        fig.canvas.mpl_connect("button_press_event", align)
        plt.show()
        transform = cv.estimateAffine2D(np.float32(pts[1]), np.float32(pts[0]))[0]

        def on_update(value: float):
            aximg.set_alpha(value)
            fig.canvas.draw_idle()

        fig, ax = plt.subplots()
        ax.imshow(img)
        img_depth = cv.warpAffine(img_depth, transform, img_depth.shape[::-1])
        aximg = ax.imshow(img_depth, cmap="gray", alpha=0.5)
        axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        fig.subplots_adjust(left=0.25, bottom=0.25)
        slider = plt.Slider(axfreq, "Blend", 0.0, 1.0)
        slider.on_changed(on_update)
        plt.show()

        return np.vstack((transform, [0, 0, 1]), dtype=np.float32)
    
    @staticmethod
    def rgb_intrinsics(path: str | Path, debug_path: str | Path | None = None):
        """
        Calibrates (finds intrinsic parameters of) the RGB module based on
        chessboard images.

        Parameters
        ------------
        path: PathLike
            Path to a directory containing chessboard images taken with the RGB module.
        debug_path: Optional[PathLike]
            Path where detected and then undistorted chessboard images will be saved.
            Used for debugging purposes to validate chessboards were properly detected
            and undistorted.
            Defaults to ``None``.
        """
        path = Path(path)
        if debug_path:
            debug_path = Path(debug_path)
            debug_path.mkdir(exist_ok=True)

        square_size = 25  # mm
        pattern_size = (9, 6)
        pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        pattern_points *= square_size

        obj_points = []
        img_points = []

        img_paths: list[Path] = []
        for filename in os.listdir(path):
            filepath = path.joinpath(filename)
            img = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
            h, w = img.shape
            s, corners = cv.findChessboardCorners(img, pattern_size)
            if not s:
                continue

            term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
            corners = cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
            img_points.append(corners)
            obj_points.append(pattern_points)
            if debug_path:
                debug_fp = debug_path.joinpath(f"corners_{filename}")
                img_paths.append(debug_fp)
                img_dbg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
                cv.drawChessboardCorners(img_dbg, pattern_size, corners, s)
                cv.imwrite(debug_fp, img_dbg)

        _, camera_matrix, distortion_coef, _, _ = cv.calibrateCamera(obj_points, img_points, (w, h), None, None)

        # undistort
        for filepath in img_paths:
            img = cv.imread(filepath)
            new_cam_mat, roi = cv.getOptimalNewCameraMatrix(camera_matrix, distortion_coef, (w, h), 1)
            img = cv.undistort(img, camera_matrix, distortion_coef, newCameraMatrix=new_cam_mat)
            cv.imwrite(filepath.parent.joinpath(f"undistorted_{filepath.name}"), img)

        return camera_matrix, distortion_coef
