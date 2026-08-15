from dataclasses import replace
from datetime import UTC, datetime

import pytest

from repvision.calibration import (
    CalibrationCollector,
    CalibrationError,
    CalibrationPosition,
    CalibrationProfile,
    CalibrationRangeError,
    CalibrationStorageError,
)
from repvision.config import AppConfig, Arm


def test_calibration_positions_have_stable_values() -> None:
    assert CalibrationPosition.EXTENDED.value == "extended"
    assert CalibrationPosition.CURLED.value == "curled"


def test_calibration_failures_share_public_base() -> None:
    assert isinstance(CalibrationRangeError("range"), CalibrationError)
    assert isinstance(CalibrationStorageError("storage"), CalibrationError)


def profile() -> CalibrationProfile:
    return CalibrationProfile(
        Arm.RIGHT,
        curled_angle=42.0,
        extended_angle=164.0,
        up_threshold=52.0,
        down_threshold=154.0,
        samples_per_position=20,
        calibrated_at=datetime(2026, 8, 15, 9, 30, tzinfo=UTC),
    )


def test_calibration_profile_keeps_personalized_range() -> None:
    result = profile()

    assert result.arm is Arm.RIGHT
    assert result.movement_range == 122.0
    assert result.samples_per_position == 20


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("curled_angle", -1.0),
        ("extended_angle", 181.0),
        ("up_threshold", float("nan")),
        ("down_threshold", float("inf")),
    ],
)
def test_calibration_profile_rejects_invalid_angles(
    field: str, value: float
) -> None:
    with pytest.raises(ValueError, match="finite and between"):
        replace(profile(), **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("curled_angle", 60.0),
        ("up_threshold", 154.0),
        ("down_threshold", 52.0),
        ("extended_angle", 154.0),
    ],
)
def test_calibration_profile_requires_ordered_endpoints(
    field: str, value: float
) -> None:
    with pytest.raises(ValueError, match="curled < up < down < extended"):
        replace(profile(), **{field: value})


def test_calibration_profile_requires_robust_sample_count() -> None:
    with pytest.raises(ValueError, match="samples_per_position"):
        replace(profile(), samples_per_position=2)


def test_calibration_profile_requires_timezone() -> None:
    with pytest.raises(ValueError, match="include a timezone"):
        replace(profile(), calibrated_at=datetime(2026, 8, 15, 9, 30))


def test_calibration_collector_retains_valid_endpoint_angles() -> None:
    collector = CalibrationCollector(AppConfig(calibration_sample_target=3), Arm.LEFT)

    assert collector.add(CalibrationPosition.EXTENDED, 160.0) == 1
    assert collector.add(CalibrationPosition.EXTENDED, 162.0) == 2
    assert collector.sample_count(CalibrationPosition.EXTENDED) == 2
    assert collector.sample_count(CalibrationPosition.CURLED) == 0


@pytest.mark.parametrize("angle", [None, float("nan"), float("inf")])
def test_calibration_collector_ignores_missing_measurements(
    angle: float | None,
) -> None:
    collector = CalibrationCollector(AppConfig(), Arm.RIGHT)

    assert collector.add(CalibrationPosition.CURLED, angle) == 0
    assert collector.sample_count(CalibrationPosition.CURLED) == 0


@pytest.mark.parametrize("angle", [-0.1, 180.1])
def test_calibration_collector_rejects_out_of_range_angle(angle: float) -> None:
    collector = CalibrationCollector(AppConfig(), Arm.RIGHT)

    with pytest.raises(ValueError, match="between 0 and 180"):
        collector.add(CalibrationPosition.EXTENDED, angle)


def test_calibration_collector_bounds_retained_history() -> None:
    collector = CalibrationCollector(AppConfig(calibration_sample_target=3), Arm.RIGHT)

    for angle in (160.0, 161.0, 162.0, 170.0):
        count = collector.add(CalibrationPosition.EXTENDED, angle)

    assert count == 3
    assert collector.sample_count(CalibrationPosition.EXTENDED) == 3
