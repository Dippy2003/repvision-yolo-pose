"""YOLO pose inference and typed keypoint extraction."""

from dataclasses import dataclass
from enum import IntEnum, StrEnum
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


class PoseStatus(StrEnum):
    """Visibility state produced by one pose inference."""

    TRACKING = "tracking"
    NO_PERSON = "no_person"
    MISSING_KEYPOINTS = "missing_keypoints"
    LOW_CONFIDENCE = "low_confidence"


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
class ArmLandmarks:
    """Shoulder, elbow, wrist, and hip landmarks for the selected arm."""

    arm: Arm
    shoulder: Landmark
    elbow: Landmark
    wrist: Landmark
    hip: Landmark

    def movement_points_reliable(self, threshold: float) -> bool:
        """Check only the three joints required for elbow movement."""
        return all(
            landmark.is_reliable(threshold)
            for landmark in (self.shoulder, self.elbow, self.wrist)
        )


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

    def arm_landmarks(self, arm: Arm) -> ArmLandmarks | None:
        """Extract one arm, or return None when any expected joint is absent."""
        selected = tuple(self.landmark(index) for index in arm_keypoint_indices(arm))
        if any(landmark is None for landmark in selected):
            return None
        shoulder, elbow, wrist, hip = selected
        assert shoulder is not None
        assert elbow is not None
        assert wrist is not None
        assert hip is not None
        return ArmLandmarks(arm, shoulder, elbow, wrist, hip)


@dataclass(frozen=True, slots=True)
class PoseObservation:
    """Structured pose information produced for one video frame."""

    persons: tuple[PersonPose, ...]
    primary_person: PersonPose | None
    selected_arm: ArmLandmarks | None
    status: PoseStatus


def select_primary_person(persons: tuple[PersonPose, ...]) -> PersonPose | None:
    """Select the largest person with a valid bounding box."""
    valid_people = (person for person in persons if person.box.area > 0.0)
    return max(valid_people, key=lambda person: person.box.area, default=None)
