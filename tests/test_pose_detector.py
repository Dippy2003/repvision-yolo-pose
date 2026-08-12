from dataclasses import FrozenInstanceError
import sys
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
    PoseDetector,
    PoseDetectorError,
    PoseInferenceError,
    PoseModel,
    PoseModelLoadError,
    PoseObservation,
    PoseResultError,
    PoseStatus,
    _as_float_array,
    _landmark_from_row,
    _result_components,
    _run_model,
    _validate_pose_shapes,
    arm_keypoint_indices,
    load_ultralytics_model,
    parse_pose_result,
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


def synthetic_result(
    boxes: NDArray[np.float64],
    confidences: NDArray[np.float64],
    keypoints: NDArray[np.float64],
) -> object:
    return SimpleNamespace(
        boxes=SimpleNamespace(xyxy=boxes, conf=confidences),
        keypoints=SimpleNamespace(data=keypoints),
    )


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
        pytest.raises(
            PoseModelLoadError, match=r"broken-pose\.pt.*invalid checkpoint"
        ),
    ):
        load_ultralytics_model("broken-pose.pt")


def test_ultralytics_loader_explains_missing_dependency() -> None:
    with (
        patch.dict(sys.modules, {"ultralytics": None}),
        pytest.raises(PoseModelLoadError, match="ultralytics"),
    ):
        load_ultralytics_model("chosen-pose.pt")


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


def test_detector_handles_inference_without_people() -> None:
    frame = np.zeros((12, 20, 3), dtype=np.uint8)
    model = RecordingModel([])
    detector = PoseDetector(AppConfig(), model_factory=lambda _name: model)

    observation = detector.detect(frame)

    assert observation == PoseObservation((), None, None, PoseStatus.NO_PERSON)
    assert len(model.calls) == 1


def test_detector_tracks_configured_arm_on_largest_person() -> None:
    frame = np.zeros((12, 20, 3), dtype=np.uint8)
    keypoints = np.zeros((2, 17, 3), dtype=np.float64)
    for person_index in range(2):
        for keypoint_index in range(17):
            keypoints[person_index, keypoint_index] = [
                float(100 * person_index + keypoint_index),
                float(200 * person_index + keypoint_index),
                0.9,
            ]
    result = synthetic_result(
        np.asarray([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 30.0, 40.0]]),
        np.asarray([0.8, 0.7]),
        keypoints,
    )
    model = RecordingModel([result])
    detector = PoseDetector(
        AppConfig(selected_arm=Arm.RIGHT), model_factory=lambda _name: model
    )

    observation = detector.detect(frame)

    assert observation.status is PoseStatus.TRACKING
    assert observation.primary_person is observation.persons[1]
    assert observation.selected_arm is not None
    assert observation.selected_arm.arm is Arm.RIGHT
    assert observation.selected_arm.shoulder.point == Point2D(106.0, 206.0)
    assert observation.selected_arm.elbow.point == Point2D(108.0, 208.0)
    assert observation.selected_arm.wrist.point == Point2D(110.0, 210.0)
    assert observation.selected_arm.hip.point == Point2D(112.0, 212.0)


def test_detector_can_override_configured_arm_for_live_switching() -> None:
    keypoints = np.zeros((1, 17, 3), dtype=np.float64)
    keypoints[0, :, 2] = 0.9
    result = synthetic_result(
        np.asarray([[0.0, 0.0, 20.0, 30.0]]),
        np.asarray([0.9]),
        keypoints,
    )
    detector = PoseDetector(
        AppConfig(selected_arm=Arm.RIGHT),
        model_factory=lambda _name: RecordingModel([result]),
    )

    observation = detector.detect(np.zeros((20, 20, 3), dtype=np.uint8), Arm.LEFT)

    assert observation.selected_arm is not None
    assert observation.selected_arm.arm is Arm.LEFT


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


def test_pose_person_counts_must_match() -> None:
    with pytest.raises(PoseResultError, match="person counts do not match"):
        _validate_pose_shapes(
            np.zeros((2, 4)), np.zeros((2,)), np.zeros((1, 17, 3))
        )


def test_empty_pose_arrays_do_not_require_landmark_rows() -> None:
    _validate_pose_shapes(np.zeros((0, 4)), np.zeros((0,)), np.zeros((0, 0, 3)))


def test_detector_reports_missing_selected_arm_keypoints() -> None:
    frame = np.zeros((12, 20, 3), dtype=np.uint8)
    keypoints = np.full((1, 10, 3), [10.0, 20.0, 0.9], dtype=np.float64)
    result = synthetic_result(
        np.asarray([[0.0, 0.0, 20.0, 30.0]]), np.asarray([0.8]), keypoints
    )
    detector = PoseDetector(
        AppConfig(selected_arm=Arm.RIGHT),
        model_factory=lambda _name: RecordingModel([result]),
    )

    observation = detector.detect(frame)

    assert observation.primary_person is not None
    assert observation.selected_arm is None
    assert observation.status is PoseStatus.MISSING_KEYPOINTS


@pytest.mark.parametrize(
    "low_confidence_index",
    [
        KeypointIndex.RIGHT_SHOULDER,
        KeypointIndex.RIGHT_ELBOW,
        KeypointIndex.RIGHT_WRIST,
    ],
)
def test_detector_rejects_unreliable_movement_keypoints(
    low_confidence_index: KeypointIndex,
) -> None:
    frame = np.zeros((12, 20, 3), dtype=np.uint8)
    keypoints = np.full((1, 17, 3), [10.0, 20.0, 0.9], dtype=np.float64)
    keypoints[0, low_confidence_index, 2] = 0.49
    result = synthetic_result(
        np.asarray([[0.0, 0.0, 20.0, 30.0]]), np.asarray([0.8]), keypoints
    )
    detector = PoseDetector(
        AppConfig(confidence_threshold=0.5, selected_arm=Arm.RIGHT),
        model_factory=lambda _name: RecordingModel([result]),
    )

    observation = detector.detect(frame)

    assert observation.selected_arm is not None
    assert observation.status is PoseStatus.LOW_CONFIDENCE


def test_detector_uses_left_side_when_configured() -> None:
    frame = np.zeros((12, 20, 3), dtype=np.uint8)
    keypoints = np.asarray(
        [[[float(i), float(i + 20), 0.9] for i in range(17)]], dtype=np.float64
    )
    result = synthetic_result(
        np.asarray([[0.0, 0.0, 20.0, 30.0]]), np.asarray([0.8]), keypoints
    )
    detector = PoseDetector(
        AppConfig(selected_arm=Arm.LEFT),
        model_factory=lambda _name: RecordingModel([result]),
    )

    observation = detector.detect(frame)

    assert observation.selected_arm is not None
    assert observation.selected_arm.arm is Arm.LEFT
    assert observation.selected_arm.shoulder.point == Point2D(5.0, 25.0)
    assert observation.selected_arm.elbow.point == Point2D(7.0, 27.0)
    assert observation.selected_arm.wrist.point == Point2D(9.0, 29.0)
    assert observation.selected_arm.hip.point == Point2D(11.0, 31.0)


def test_landmark_row_preserves_coordinates_and_confidence() -> None:
    landmark = _landmark_from_row(np.asarray([14.0, 28.0, 0.75]))

    assert landmark == Landmark(Point2D(14.0, 28.0), 0.75)


@pytest.mark.parametrize(
    "row",
    [
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([float("nan"), 20.0, 0.8]),
        np.asarray([10.0, 20.0, float("nan")]),
    ],
)
def test_missing_or_nonfinite_keypoints_have_no_position(
    row: NDArray[np.float64],
) -> None:
    assert _landmark_from_row(row).point is None


def test_pose_result_converts_boxes_confidence_and_keypoints() -> None:
    landmark_data = np.asarray(
        [[[float(i), float(i + 10), 0.8] for i in range(17)]], dtype=np.float64
    )
    result = synthetic_result(
        np.asarray([[10.0, 20.0, 110.0, 220.0]]),
        np.asarray([0.91]),
        landmark_data,
    )

    people = parse_pose_result(result)

    assert len(people) == 1
    assert people[0].box == BoundingBox(10.0, 20.0, 110.0, 220.0)
    assert people[0].detection_confidence == 0.91
    assert people[0].landmarks[8] == Landmark(Point2D(8.0, 18.0), 0.8)


def test_empty_pose_result_returns_no_people() -> None:
    result = synthetic_result(
        np.zeros((0, 4)), np.zeros((0,)), np.zeros((0, 17, 3))
    )

    assert parse_pose_result(result) == ()


def test_pose_result_preserves_person_box_keypoint_pairing() -> None:
    keypoints = np.zeros((2, 17, 3), dtype=np.float64)
    keypoints[0, :, :] = [10.0, 20.0, 0.7]
    keypoints[1, :, :] = [30.0, 40.0, 0.9]
    result = synthetic_result(
        np.asarray([[0.0, 0.0, 20.0, 20.0], [5.0, 5.0, 55.0, 65.0]]),
        np.asarray([0.6, 0.95]),
        keypoints,
    )

    first, second = parse_pose_result(result)

    assert first.detection_confidence == 0.6
    assert first.landmarks[5].point == Point2D(10.0, 20.0)
    assert second.detection_confidence == 0.95
    assert second.landmarks[5].point == Point2D(30.0, 40.0)


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
