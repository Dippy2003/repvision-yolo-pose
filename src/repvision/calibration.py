"""Personal arm-range calibration and local profile persistence."""

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite
from statistics import median

from repvision.config import AppConfig, Arm

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


class CalibrationCollector:
    """Collect bounded valid endpoint angles before deriving a profile."""

    def __init__(self, config: AppConfig, arm: Arm) -> None:
        self.config = config
        self.arm = arm
        self._samples: dict[CalibrationPosition, list[float]] = {
            CalibrationPosition.EXTENDED: [],
            CalibrationPosition.CURLED: [],
        }

    def add(self, position: CalibrationPosition, angle: float | None) -> int:
        """Add one usable angle and return that endpoint's retained count."""
        samples = self._samples[position]
        if angle is None or not isfinite(angle):
            return len(samples)
        if not 0.0 <= angle <= 180.0:
            raise ValueError("calibration angle must be between 0 and 180")
        if len(samples) < self.config.calibration_sample_target:
            samples.append(float(angle))
        return len(samples)

    def sample_count(self, position: CalibrationPosition) -> int:
        """Return retained valid samples for one endpoint."""
        return len(self._samples[position])

    def position_ready(self, position: CalibrationPosition) -> bool:
        """Return whether one endpoint has the configured number of samples."""
        return self.sample_count(position) >= self.config.calibration_sample_target

    @property
    def complete(self) -> bool:
        """Return whether both endpoint positions are ready."""
        return all(self.position_ready(position) for position in CalibrationPosition)

    def build_profile(
        self, calibrated_at: datetime | None = None
    ) -> CalibrationProfile:
        """Derive conservative thresholds from robust endpoint medians."""
        if not self.complete:
            raise CalibrationError(
                "Calibration requires complete extended and curled samples."
            )
        extended = float(median(self._samples[CalibrationPosition.EXTENDED]))
        curled = float(median(self._samples[CalibrationPosition.CURLED]))
        movement_range = extended - curled
        if movement_range < self.config.calibration_minimum_range:
            raise CalibrationRangeError(
                "Calibration movement range is too small "
                f"({movement_range:.1f} degrees; minimum "
                f"{self.config.calibration_minimum_range:.1f})."
            )
        margin = self.config.calibration_threshold_margin
        return CalibrationProfile(
            self.arm,
            curled,
            extended,
            curled + margin,
            extended - margin,
            self.config.calibration_sample_target,
            datetime.now(UTC) if calibrated_at is None else calibrated_at,
        )

    def reset(self, position: CalibrationPosition | None = None) -> None:
        """Clear one endpoint or restart all calibration samples."""
        positions = tuple(CalibrationPosition) if position is None else (position,)
        for selected in positions:
            self._samples[selected].clear()


def profile_to_dict(profile: CalibrationProfile) -> dict[str, object]:
    """Serialize one profile without any frame or raw keypoint data."""
    return {
        "arm": profile.arm.value,
        "curled_angle": profile.curled_angle,
        "extended_angle": profile.extended_angle,
        "up_threshold": profile.up_threshold,
        "down_threshold": profile.down_threshold,
        "samples_per_position": profile.samples_per_position,
        "calibrated_at": profile.calibrated_at.isoformat(),
    }
