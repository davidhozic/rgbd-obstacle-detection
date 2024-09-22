from __future__ import annotations
from collections import deque

import numpy as np
import cv2 as cv


__all__ = ("DepthCamera",)


class DepthCamera:
    """
    Used for processing depth images.
    """
    def __init__(self, n_frames: int, affine: np.ndarray) -> None:
        # Remove homogeneous component
        self.samples = deque([], maxlen=n_frames)
        affine = np.matrix(affine)
        self.inv_affine = affine.I
        self.affine = affine[:2]

    def align(self, img: np.ndarray):
        """
        Aligns the ``img`` (depth image) with affine transform.

        Parameters
        ------------
        img: np.ndarray
            The depth image to align.
        """
        return cv.warpAffine(img, self.affine, img.shape[::-1])

    def sample(self, img: np.ndarray):
        """
        Adds a sample to the running mean.
        """
        self.samples.append(img)

    def fill_holes(self, img: np.ndarray):
        """
        Fills in missing values (holes) by using a running mean.
        """
        samples = self.samples
        if len(samples) < 2:
            return img

        mask_is_zero = img == 0
        valid_samples = []
        for sample in samples:
            valid_samples.append(sample[mask_is_zero])       

        img[mask_is_zero] = np.max(valid_samples, axis=0)
        return img

    @staticmethod
    def normalize(img: np.ndarray, histogram: bool = True):
        """
        Performs min-max normalization as well as histogram equalization.
        """
        img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        if histogram:
            img = cv.equalizeHist(img)

        return img
