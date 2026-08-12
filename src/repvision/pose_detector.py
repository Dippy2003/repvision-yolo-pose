"""YOLO pose inference and typed keypoint extraction."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from math import isfinite
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from repvision.camera import Frame
from repvision.config import AppConfig, Arm


class PoseDetectorError(RuntimeError):
    """Base class for expected pose detector failures."""


class PoseModelLoadError(PoseDetectorError):
    """Raised when the configured pose model cannot be loaded."""


class PoseInferenceError(PoseDetectorError):
    """Raised when pose inference fails for a frame."""


class PoseResultError(PoseDetectorError):
    """Raised when model output does not match the expected pose schema."""


class PoseModel(Protocol):
    """Inference surface required from an Ultralytics pose model."""

    def predict(
        self, *, source: Frame, imgsz: int, verbose: bool
    ) -> Sequence[object]: ...


PoseModelFactory = Callable[[str], PoseModel]


def load_ultralytics_model(model_name: str) -> PoseModel:
    """Load the configured Ultralytics model on first detector construction."""
    try:
        from ultralytics import YOLO

        return YOLO(model_name)
    except (
        ConnectionError,
        FileNotFoundError,
        ImportError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        raise PoseModelLoadError(
            f"Could not load pose model '{model_name}': {error}"
        ) from error


def _run_model(model: PoseModel, frame: Frame, input_size: int) -> Sequence[object]:
    try:
        return model.predict(source=frame, imgsz=input_size, verbose=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise PoseInferenceError(f"Pose inference failed: {error}") from error


def _as_float_array(value: object) -> NDArray[np.float64]:
    cpu_method = getattr(value, "cpu", None)
    if callable(cpu_method):
        value = cpu_method()
    numpy_method = getattr(value, "numpy", None)
    if callable(numpy_method):
        value = numpy_method()
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise PoseResultError("Pose result contains non-numeric values.") from error


def _result_components(result: object) -> tuple[object, object, object] | None:
    boxes = getattr(result, "boxes", None)
    keypoints = getattr(result, "keypoints", None)
    if boxes is None or keypoints is None:
        return None

    coordinates = getattr(boxes, "xyxy", None)
    confidences = getattr(boxes, "conf", None)
    landmark_data = getattr(keypoints, "data", None)
    if coordinates is None or confidences is None or landmark_data is None:
        return None
    return coordinates, confidences, landmark_data


def _validate_pose_shapes(
    boxes: NDArray[np.float64],
    confidences: NDArray[np.float64],
    keypoints: NDArray[np.float64],
) -> None:
    if boxes.ndim != 2 or boxes.shape[1] < 4:
        raise PoseResultError("Pose boxes must have shape (people, 4 or more).")
    if confidences.ndim != 1:
        raise PoseResultError("Box confidence values must be one-dimensional.")
    if keypoints.ndim != 3 or keypoints.shape[2] < 3:
        raise PoseResultError(
            "Pose keypoints must have shape (people, keypoints, 3 or more)."
        )
    person_counts = {boxes.shape[0], confidences.shape[0], keypoints.shape[0]}
    if len(person_counts) != 1:
        raise PoseResultError("Box and keypoint person counts do not match.")


def _landmark_from_row(row: NDArray[np.float64]) -> Landmark:
    x, y, confidence = (float(value) for value in row[:3])
    if not isfinite(confidence):
        confidence = 0.0
    point = None
    if confidence > 0.0 and isfinite(x) and isfinite(y):
        point = Point2D(x, y)
    return Landmark(point, confidence)


def parse_pose_result(result: object) -> tuple[PersonPose, ...]:
    """Convert one Ultralytics result into application-owned person poses."""
    components = _result_components(result)
    if components is None:
        return ()

    raw_boxes, raw_confidences, raw_keypoints = components
    boxes = _as_float_array(raw_boxes)
    confidences = _as_float_array(raw_confidences)
    keypoints = _as_float_array(raw_keypoints)
    _validate_pose_shapes(boxes, confidences, keypoints)

    people: list[PersonPose] = []
    for box_row, confidence, landmark_rows in zip(
        boxes, confidences, keypoints, strict=True
    ):
        box = BoundingBox(*(float(value) for value in box_row[:4]))
        landmarks = tuple(_landmark_from_row(row) for row in landmark_rows)
        people.append(PersonPose(box, landmarks, float(confidence)))
    return tuple(people)


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
    """Select the largest valid person, preserving model order for area ties."""
    valid_people = (person for person in persons if person.box.area > 0.0)
    return max(valid_people, key=lambda person: person.box.area, default=None)


class PoseDetector:
    """Run one configured pose model and return application-owned data."""

    def __init__(
        self,
        config: AppConfig,
        model_factory: PoseModelFactory = load_ultralytics_model,
    ) -> None:
        self.config = config
        self._model = model_factory(config.model_name)

    def detect(self, frame: Frame, arm: Arm | None = None) -> PoseObservation:
        """Run pose inference and extract the requested arm from one person."""
        results = _run_model(self._model, frame, self.config.input_size)
        persons = parse_pose_result(results[0]) if results else ()
        primary_person = select_primary_person(persons)
        if primary_person is None:
            return PoseObservation(persons, None, None, PoseStatus.NO_PERSON)

        selected_arm = primary_person.arm_landmarks(arm or self.config.selected_arm)
        if selected_arm is None:
            status = PoseStatus.MISSING_KEYPOINTS
        elif not selected_arm.movement_points_reliable(
            self.config.confidence_threshold
        ):
            status = PoseStatus.LOW_CONFIDENCE
        else:
            status = PoseStatus.TRACKING
        return PoseObservation(persons, primary_person, selected_arm, status)
