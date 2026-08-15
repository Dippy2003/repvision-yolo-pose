from dataclasses import replace
from datetime import UTC, datetime

import pytest

from repvision.calibration import (
    CalibrationError,
    CalibrationPosition,
    CalibrationProfile,
    CalibrationRangeError,
    CalibrationStorageError,
)
from repvision.config import Arm


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
