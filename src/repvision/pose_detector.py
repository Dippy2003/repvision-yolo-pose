"""YOLO pose inference and typed keypoint extraction."""

from dataclasses import dataclass
from enum import IntEnum


class KeypointIndex(IntEnum):
    """COCO human-pose indices used by RepVision."""

    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12


@dataclass(frozen=True, slots=True)
class Point2D:
    """Pixel coordinate in an input frame."""

    x: float
    y: float
