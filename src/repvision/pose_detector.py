"""YOLO pose inference and typed keypoint extraction."""

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
