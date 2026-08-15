"""Personal arm-range calibration and local profile persistence."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from math import isfinite

from repvision.config import Arm

CALIBRATION_SCHEMA_VERSION = 1


class CalibrationError(RuntimeError):
    """Base class for expected calibration failures."""


class CalibrationRangeError(CalibrationError):
    """Raised when captured positions do not define enough movement range."""


class CalibrationStorageError(CalibrationError):
    """Raised when calibration profiles cannot be loaded or saved safely."""


class CalibrationPosition(StrEnum):
    """Arm endpoints captured during calibration."""

    EXTENDED = "extended"
    CURLED = "curled"


@dataclass(frozen=True, slots=True)
class CalibrationProfile:
    """Validated personalized movement endpoints for one arm."""

    arm: Arm
    curled_angle: float
    extended_angle: float
    up_threshold: float
    down_threshold: float
    samples_per_position: int
    calibrated_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.arm, Arm):
            raise ValueError("arm must be left or right")
        angles = (
            self.curled_angle,
            self.up_threshold,
            self.down_threshold,
            self.extended_angle,
        )
        if any(not isfinite(angle) or not 0.0 <= angle <= 180.0 for angle in angles):
            raise ValueError("calibration angles must be finite and between 0 and 180")
        if not (
            self.curled_angle
            < self.up_threshold
            < self.down_threshold
            < self.extended_angle
        ):
            raise ValueError(
                "calibration angles must satisfy curled < up < down < extended"
            )
        if self.samples_per_position < 3:
            raise ValueError("samples_per_position must be at least 3")
        if self.calibrated_at.utcoffset() is None:
            raise ValueError("calibrated_at must include a timezone")

    @property
    def movement_range(self) -> float:
        """Return the measured endpoint range in degrees."""
        return self.extended_angle - self.curled_angle
