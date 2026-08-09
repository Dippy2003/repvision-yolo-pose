from dataclasses import FrozenInstanceError

import pytest

from repvision.config import Arm
from repvision.pose_detector import (
    ArmLandmarks,
    BoundingBox,
    KeypointIndex,
    Landmark,
    PersonPose,
    Point2D,
    PoseStatus,
    arm_keypoint_indices,
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
