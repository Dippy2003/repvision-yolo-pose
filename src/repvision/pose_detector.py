"""YOLO pose inference and typed keypoint extraction."""

from dataclasses import dataclass
from enum import IntEnum
from math import isfinite

from repvision.config import Arm


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


def arm_keypoint_indices(
    arm: Arm,
) -> tuple[KeypointIndex, KeypointIndex, KeypointIndex, KeypointIndex]:
    """Return shoulder, elbow, wrist, and hip indices for one arm."""
    if arm is Arm.LEFT:
        return (
            KeypointIndex.LEFT_SHOULDER,
            KeypointIndex.LEFT_ELBOW,
            KeypointIndex.LEFT_WRIST,
            KeypointIndex.LEFT_HIP,
        )
    return (
        KeypointIndex.RIGHT_SHOULDER,
        KeypointIndex.RIGHT_ELBOW,
        KeypointIndex.RIGHT_WRIST,
        KeypointIndex.RIGHT_HIP,
    )


@dataclass(frozen=True, slots=True)
class Point2D:
    """Pixel coordinate in an input frame."""

    x: float
    y: float


@dataclass(frozen=True, slots=True)
class Landmark:
    """Pose point with its model-reported confidence."""

    point: Point2D | None
    confidence: float

    def is_reliable(self, threshold: float) -> bool:
        """Return whether this landmark is present and meets a threshold."""
        return self.point is not None and self.confidence >= threshold


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Person bounding box in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        """Return zero for degenerate or non-finite boxes."""
        coordinates = (self.x1, self.y1, self.x2, self.y2)
        if not all(isfinite(value) for value in coordinates):
            return 0.0
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass(frozen=True, slots=True)
class PersonPose:
    """One detected person's box and ordered COCO landmarks."""

    box: BoundingBox
    landmarks: tuple[Landmark, ...]
    detection_confidence: float

    def landmark(self, index: KeypointIndex) -> Landmark | None:
        """Return a landmark when the model supplied the requested index."""
        if index >= len(self.landmarks):
            return None
        return self.landmarks[index]
