from math import sqrt

import numpy as np
import pytest

from repvision.angles import (
    AngleSmoother,
    _clamp_cosine,
    _coordinates,
    calculate_arm_angle,
    calculate_elbow_angle,
)
from repvision.config import Arm
from repvision.pose_detector import ArmLandmarks, Landmark, Point2D


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ((1, 2), (1.0, 2.0)),
        ([1.5, 2.5], (1.5, 2.5)),
        (np.asarray([3, 4]), (3.0, 4.0)),
    ],
)
def test_coordinates_accept_numeric_2d_inputs(
    point: object, expected: tuple[float, float]
) -> None:
    assert _coordinates(point) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize("point", [(), (1,), (1, 2, 3), ("x", 2), None])
def test_coordinates_reject_invalid_inputs(point: object) -> None:
    with pytest.raises(ValueError, match="exactly two numeric"):
        _coordinates(point)  # type: ignore[arg-type]


def test_straight_arm_angle_is_180_degrees() -> None:
    angle = calculate_elbow_angle((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))

    assert angle == pytest.approx(180.0)


@pytest.mark.parametrize(
    ("shoulder", "elbow", "wrist"),
    [
        ((0.0, 1.0), (0.0, 0.0), (1.0, 0.0)),
        ((4.0, 3.0), (2.0, 3.0), (2.0, 8.0)),
    ],
)
def test_perpendicular_arm_angle_is_90_degrees(
    shoulder: tuple[float, float],
    elbow: tuple[float, float],
    wrist: tuple[float, float],
) -> None:
    assert calculate_elbow_angle(shoulder, elbow, wrist) == pytest.approx(90.0)


def test_bent_arm_angle_matches_known_45_degree_geometry() -> None:
    diagonal = sqrt(0.5)

    angle = calculate_elbow_angle(
        (1.0, 0.0), (0.0, 0.0), (diagonal, diagonal)
    )

    assert angle == pytest.approx(45.0)


@pytest.mark.parametrize(
    ("shoulder", "elbow", "wrist"),
    [
        ((1.0, 1.0), (1.0, 1.0), (2.0, 1.0)),
        ((0.0, 1.0), (1.0, 1.0), (1.0, 1.0)),
        ((1.0, 1.0), (1.0, 1.0), (1.0, 1.0)),
    ],
)
def test_zero_length_limb_vectors_return_no_angle(
    shoulder: tuple[float, float],
    elbow: tuple[float, float],
    wrist: tuple[float, float],
) -> None:
    assert calculate_elbow_angle(shoulder, elbow, wrist) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [(-1.0000000001, -1.0), (-0.5, -0.5), (0.5, 0.5), (1.0000000001, 1.0)],
)
def test_cosine_is_clamped_for_floating_point_safety(
    value: float, expected: float
) -> None:
    assert _clamp_cosine(value) == expected


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_coordinates_return_no_angle(invalid: float) -> None:
    assert calculate_elbow_angle((invalid, 0.0), (1.0, 0.0), (2.0, 0.0)) is None


@pytest.mark.parametrize("window_size", [0, -1])
def test_smoother_requires_positive_window_size(window_size: int) -> None:
    with pytest.raises(ValueError, match="window_size must be positive"):
        AngleSmoother(window_size)


def test_new_smoother_has_no_samples() -> None:
    smoother = AngleSmoother(window_size=5)

    assert smoother.sample_count == 0
    assert smoother.value is None


def test_smoother_uses_median_to_reject_single_outlier() -> None:
    smoother = AngleSmoother(window_size=5)

    values = [smoother.add(angle) for angle in (160.0, 158.0, 40.0)]

    assert values == [160.0, 159.0, 158.0]
    assert smoother.sample_count == 3


def test_smoother_discards_values_outside_window() -> None:
    smoother = AngleSmoother(window_size=3)
    for angle in (10.0, 20.0, 30.0, 100.0):
        smoother.add(angle)

    assert smoother.sample_count == 3
    assert smoother.value == 30.0


@pytest.mark.parametrize("missing", [None, float("nan"), float("inf")])
def test_smoother_ignores_missing_or_nonfinite_values(
    missing: float | None,
) -> None:
    smoother = AngleSmoother(window_size=3)
    smoother.add(90.0)

    assert smoother.add(missing) is None
    assert smoother.sample_count == 1
    assert smoother.value == 90.0


@pytest.mark.parametrize("invalid", [-0.1, 180.1])
def test_smoother_rejects_out_of_range_angles(invalid: float) -> None:
    smoother = AngleSmoother(window_size=3)

    with pytest.raises(ValueError, match="between 0 and 180"):
        smoother.add(invalid)

    assert smoother.sample_count == 0


def test_smoother_reset_clears_history() -> None:
    smoother = AngleSmoother(window_size=3)
    smoother.add(60.0)
    smoother.add(70.0)

    smoother.reset()

    assert smoother.sample_count == 0
    assert smoother.value is None


def test_selected_arm_angle_requires_reliable_movement_points() -> None:
    shoulder = Landmark(Point2D(0.0, 1.0), 0.9)
    elbow = Landmark(Point2D(0.0, 0.0), 0.9)
    wrist = Landmark(Point2D(1.0, 0.0), 0.9)
    hip = Landmark(Point2D(0.0, 2.0), 0.2)
    reliable = ArmLandmarks(Arm.RIGHT, shoulder, elbow, wrist, hip)
    low_wrist = ArmLandmarks(
        Arm.RIGHT, shoulder, elbow, Landmark(wrist.point, 0.4), hip
    )

    assert calculate_arm_angle(reliable, 0.5) == pytest.approx(90.0)
    assert calculate_arm_angle(low_wrist, 0.5) is None
    assert calculate_arm_angle(None, 0.5) is None
