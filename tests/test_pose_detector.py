from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

from repvision.config import AppConfig, Arm
from repvision.pose_detector import (
    ArmLandmarks,
    BoundingBox,
    KeypointIndex,
    Landmark,
    PersonPose,
    Point2D,
    PoseObservation,
    PoseDetectorError,
    PoseDetector,
    PoseInferenceError,
    PoseModel,
    PoseModelLoadError,
    PoseResultError,
    PoseStatus,
    _as_float_array,
    _result_components,
    _run_model,
    _validate_pose_shapes,
    arm_keypoint_indices,
    load_ultralytics_model,
    select_primary_person,
)


class RecordingModel:
    def __init__(self, results: list[object] | None = None) -> None:
        self.results = results or []
        self.calls: list[tuple[NDArray[np.uint8], int, bool]] = []

    def predict(
        self, *, source: NDArray[np.uint8], imgsz: int, verbose: bool
    ) -> list[object]:
        self.calls.append((source, imgsz, verbose))
        return self.results


class FailingModel:
    def predict(
        self, *, source: NDArray[np.uint8], imgsz: int, verbose: bool
    ) -> list[object]:
        raise RuntimeError("backend unavailable")


class TensorLike:
    def __init__(self, values: list[list[float]]) -> None:
        self.values = values
        self.cpu_calls = 0

    def cpu(self) -> "TensorLike":
        self.cpu_calls += 1
        return self

    def numpy(self) -> NDArray[np.float64]:
        return np.asarray(self.values, dtype=np.float64)


def test_coco_arm_keypoint_indices_match_model_schema() -> None:
    assert KeypointIndex.LEFT_SHOULDER == 5
    assert KeypointIndex.RIGHT_SHOULDER == 6
    assert KeypointIndex.LEFT_ELBOW == 7
    assert KeypointIndex.RIGHT_ELBOW == 8
    assert KeypointIndex.LEFT_WRIST == 9
    assert KeypointIndex.RIGHT_WRIST == 10
    assert KeypointIndex.LEFT_HIP == 11
    assert KeypointIndex.RIGHT_HIP == 12


def test_pose_status_values_are_stable_for_consumers() -> None:
    assert PoseStatus.TRACKING.value == "tracking"
    assert PoseStatus.NO_PERSON.value == "no_person"
    assert PoseStatus.MISSING_KEYPOINTS.value == "missing_keypoints"
    assert PoseStatus.LOW_CONFIDENCE.value == "low_confidence"


@pytest.mark.parametrize(
    "error_type", [PoseModelLoadError, PoseInferenceError, PoseResultError]
)
def test_pose_failures_share_a_public_base_error(
    error_type: type[PoseDetectorError],
) -> None:
    assert isinstance(error_type("failure"), PoseDetectorError)


def test_ultralytics_loader_forwards_configured_model_name() -> None:
    sentinel_model = object()
    with patch("ultralytics.YOLO", return_value=sentinel_model) as yolo:
        loaded = load_ultralytics_model("chosen-pose.pt")

    assert loaded is sentinel_model
    yolo.assert_called_once_with("chosen-pose.pt")


def test_ultralytics_loader_explains_model_failure() -> None:
    with (
        patch("ultralytics.YOLO", side_effect=RuntimeError("invalid checkpoint")),
        pytest.raises(PoseModelLoadError, match="broken-pose.pt.*invalid checkpoint"),
    ):
        load_ultralytics_model("broken-pose.pt")


def test_detector_loads_configured_model_once() -> None:
    loaded_names: list[str] = []
    sentinel_model = cast(PoseModel, object())

    detector = PoseDetector(
        AppConfig(model_name="chosen-pose.pt"),
        model_factory=lambda name: loaded_names.append(name) or sentinel_model,
    )

    assert detector.config.model_name == "chosen-pose.pt"
    assert loaded_names == ["chosen-pose.pt"]


def test_model_inference_receives_frame_and_configured_size() -> None:
    frame = np.zeros((12, 20, 3), dtype=np.uint8)
    model = RecordingModel([object()])

    results = _run_model(model, frame, 320)

    assert results is model.results
    assert len(model.calls) == 1
    called_frame, input_size, verbose = model.calls[0]
    assert called_frame is frame
    assert input_size == 320
    assert verbose is False


def test_model_inference_wraps_expected_backend_failures() -> None:
    frame = np.zeros((12, 20, 3), dtype=np.uint8)

    with pytest.raises(PoseInferenceError, match="backend unavailable"):
        _run_model(FailingModel(), frame, 640)


def test_tensor_output_is_moved_to_cpu_and_converted_to_float_array() -> None:
    tensor = TensorLike([[1.0, 2.0], [3.0, 4.0]])

    converted = _as_float_array(tensor)

    assert tensor.cpu_calls == 1
    assert converted.dtype == np.float64
    np.testing.assert_array_equal(converted, tensor.values)


def test_non_numeric_model_output_is_rejected() -> None:
    with pytest.raises(PoseResultError, match="non-numeric"):
        _as_float_array([["not-a-number"]])


@pytest.mark.parametrize(
    "result",
    [
        SimpleNamespace(boxes=None, keypoints=None),
        SimpleNamespace(boxes=SimpleNamespace(), keypoints=SimpleNamespace()),
        SimpleNamespace(boxes=SimpleNamespace(xyxy=[]), keypoints=None),
    ],
)
def test_result_components_handle_frames_without_pose_data(result: object) -> None:
    assert _result_components(result) is None


@pytest.mark.parametrize(
    ("boxes", "confidences", "keypoints", "message"),
    [
        (np.zeros((4,)), np.zeros((1,)), np.zeros((1, 17, 3)), "Pose boxes"),
        (np.zeros((1, 4)), np.zeros((1, 1)), np.zeros((1, 17, 3)), "confidence"),
        (np.zeros((1, 4)), np.zeros((1,)), np.zeros((17, 3)), "Pose keypoints"),
        (np.zeros((1, 4)), np.zeros((1,)), np.zeros((1, 17, 2)), "Pose keypoints"),
    ],
)
def test_pose_array_shapes_are_validated(
    boxes: NDArray[np.float64],
    confidences: NDArray[np.float64],
    keypoints: NDArray[np.float64],
    message: str,
) -> None:
    with pytest.raises(PoseResultError, match=message):
        _validate_pose_shapes(boxes, confidences, keypoints)


def test_empty_observation_represents_a_frame_without_people() -> None:
    observation = PoseObservation((), None, None, PoseStatus.NO_PERSON)

    assert observation.persons == ()
    assert observation.primary_person is None
    assert observation.selected_arm is None


def test_primary_person_is_largest_valid_detection() -> None:
    landmarks = tuple(Landmark(Point2D(1.0, 1.0), 0.9) for _ in range(17))
    invalid = PersonPose(BoundingBox(5.0, 5.0, 2.0, 8.0), landmarks, 0.99)
    small = PersonPose(BoundingBox(0.0, 0.0, 10.0, 10.0), landmarks, 0.9)
    large = PersonPose(BoundingBox(0.0, 0.0, 20.0, 15.0), landmarks, 0.6)

    assert select_primary_person((invalid, small, large)) is large
    assert select_primary_person((invalid,)) is None
    assert select_primary_person(()) is None


def test_primary_person_selection_is_stable_for_equal_areas() -> None:
    landmarks = tuple(Landmark(Point2D(1.0, 1.0), 0.9) for _ in range(17))
    first = PersonPose(BoundingBox(0.0, 0.0, 10.0, 20.0), landmarks, 0.6)
    second = PersonPose(BoundingBox(5.0, 5.0, 25.0, 15.0), landmarks, 0.9)

    assert select_primary_person((first, second)) is first


def test_point_coordinates_are_immutable() -> None:
    point = Point2D(x=12.5, y=24.0)

    assert (point.x, point.y) == (12.5, 24.0)
    with pytest.raises(FrozenInstanceError):
        point.x = 9.0  # type: ignore[misc]


def test_landmark_reliability_requires_position_and_confidence() -> None:
    point = Point2D(10.0, 20.0)

    assert Landmark(point, 0.5).is_reliable(0.5)
    assert not Landmark(point, 0.49).is_reliable(0.5)
    assert not Landmark(None, 0.99).is_reliable(0.5)


@pytest.mark.parametrize(
    ("box", "area"),
    [
        (BoundingBox(10.0, 20.0, 30.0, 50.0), 600.0),
        (BoundingBox(30.0, 20.0, 10.0, 50.0), 0.0),
        (BoundingBox(10.0, 50.0, 30.0, 20.0), 0.0),
        (BoundingBox(0.0, 0.0, float("nan"), 10.0), 0.0),
    ],
)
def test_bounding_box_area_rejects_invalid_geometry(
    box: BoundingBox, area: float
) -> None:
    assert box.area == area


def test_person_pose_handles_missing_keypoint_indices() -> None:
    landmarks = tuple(Landmark(Point2D(float(i), float(i)), 0.9) for i in range(8))
    person = PersonPose(BoundingBox(0.0, 0.0, 10.0, 20.0), landmarks, 0.8)

    assert person.landmark(KeypointIndex.LEFT_ELBOW) == landmarks[7]
    assert person.landmark(KeypointIndex.LEFT_WRIST) is None


def test_person_pose_extracts_selected_side_landmarks() -> None:
    landmarks = tuple(
        Landmark(Point2D(float(i), float(i + 1)), 0.9) for i in range(17)
    )
    person = PersonPose(BoundingBox(0.0, 0.0, 10.0, 20.0), landmarks, 0.8)

    left = person.arm_landmarks(Arm.LEFT)
    right = person.arm_landmarks(Arm.RIGHT)

    assert left is not None and left.shoulder == landmarks[5]
    assert left.elbow == landmarks[7]
    assert left.wrist == landmarks[9]
    assert left.hip == landmarks[11]
    assert right is not None and right.shoulder == landmarks[6]
    assert right.elbow == landmarks[8]
    assert right.wrist == landmarks[10]
    assert right.hip == landmarks[12]


def test_person_pose_returns_no_arm_when_keypoints_are_missing() -> None:
    landmarks = tuple(
        Landmark(Point2D(float(i), float(i)), 0.9) for i in range(10)
    )
    person = PersonPose(BoundingBox(0.0, 0.0, 10.0, 20.0), landmarks, 0.8)

    assert person.arm_landmarks(Arm.RIGHT) is None


def test_arm_selection_maps_to_matching_coco_side() -> None:
    assert arm_keypoint_indices(Arm.LEFT) == (
        KeypointIndex.LEFT_SHOULDER,
        KeypointIndex.LEFT_ELBOW,
        KeypointIndex.LEFT_WRIST,
        KeypointIndex.LEFT_HIP,
    )
    assert arm_keypoint_indices(Arm.RIGHT) == (
        KeypointIndex.RIGHT_SHOULDER,
        KeypointIndex.RIGHT_ELBOW,
        KeypointIndex.RIGHT_WRIST,
        KeypointIndex.RIGHT_HIP,
    )


def test_arm_reliability_requires_shoulder_elbow_and_wrist() -> None:
    visible = Landmark(Point2D(1.0, 2.0), 0.8)
    low_confidence = Landmark(Point2D(2.0, 3.0), 0.2)
    landmarks = ArmLandmarks(Arm.RIGHT, visible, visible, low_confidence, visible)

    assert not landmarks.movement_points_reliable(0.5)


def test_low_confidence_hip_does_not_hide_reliable_movement_points() -> None:
    visible = Landmark(Point2D(1.0, 2.0), 0.8)
    missing_hip = Landmark(None, 0.0)
    landmarks = ArmLandmarks(Arm.LEFT, visible, visible, visible, missing_hip)

    assert landmarks.movement_points_reliable(0.5)
