from dataclasses import FrozenInstanceError

import pytest

from repvision.pose_detector import KeypointIndex, Landmark, Point2D


def test_coco_arm_keypoint_indices_match_model_schema() -> None:
    assert KeypointIndex.LEFT_SHOULDER == 5
    assert KeypointIndex.RIGHT_SHOULDER == 6
    assert KeypointIndex.LEFT_ELBOW == 7
    assert KeypointIndex.RIGHT_ELBOW == 8
    assert KeypointIndex.LEFT_WRIST == 9
    assert KeypointIndex.RIGHT_WRIST == 10
    assert KeypointIndex.LEFT_HIP == 11
    assert KeypointIndex.RIGHT_HIP == 12


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
