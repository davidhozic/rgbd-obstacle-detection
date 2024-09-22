from __future__ import annotations
from dataclasses import dataclass
import numpy as np


__all__ = (
    "ObstacleResult",
    "DoorResult",
    "ElevatorResult",
    "VerticalDoorResult"
)


@dataclass
class ObstacleResult:
    pos: np.ndarray
    width: float
    height: float
    id: int

@dataclass
class DoorResult:
    ids = [-1, -1]  # [opened id, closed id]
    opened: bool

class ElevatorResult(DoorResult):
    ids = [0, 1]

class VerticalDoorResult(DoorResult):
    ids = [2, 3]
