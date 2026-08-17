"""Personal arm-range calibration and local profile persistence."""

import json
import os
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite
from pathlib import Path
from statistics import median
from tempfile import NamedTemporaryFile

from repvision.config import AppConfig, Arm

CALIBRATION_SCHEMA_VERSION = 1


def default_calibration_path(
    environment: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    """Return a user-local configuration path without creating it."""
    values = os.environ if environment is None else environment
    local_app_data = values.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "RepVision" / "calibration.json"
    xdg_config = values.get("XDG_CONFIG_HOME")
    if xdg_config:
        return Path(xdg_config) / "repvision" / "calibration.json"
    home_directory = Path.home() if home is None else home
    return home_directory / ".config" / "repvision" / "calibration.json"


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


class CalibrationStage(StrEnum):
    """User-visible state of the guided calibration workflow."""

    READY_EXTENDED = "ready_extended"
    CAPTURING_EXTENDED = "capturing_extended"
    READY_CURLED = "ready_curled"
    CAPTURING_CURLED = "capturing_curled"
    COMPLETE = "complete"


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


class GuidedCalibration:
    """Coordinate explicit endpoint capture without camera dependencies."""

    def __init__(self, config: AppConfig, arm: Arm) -> None:
        self.collector = CalibrationCollector(config, arm)
        self.stage = CalibrationStage.READY_EXTENDED

    @property
    def arm(self) -> Arm:
        """Return the arm being calibrated."""
        return self.collector.arm

    def begin_capture(self) -> None:
        """Start sampling the endpoint currently requested from the user."""
        transitions = {
            CalibrationStage.READY_EXTENDED: CalibrationStage.CAPTURING_EXTENDED,
            CalibrationStage.READY_CURLED: CalibrationStage.CAPTURING_CURLED,
        }
        if self.stage in transitions:
            self.stage = transitions[self.stage]

    @property
    def position(self) -> CalibrationPosition | None:
        """Return the endpoint associated with the current stage."""
        if self.stage in (
            CalibrationStage.READY_EXTENDED,
            CalibrationStage.CAPTURING_EXTENDED,
        ):
            return CalibrationPosition.EXTENDED
        if self.stage in (
            CalibrationStage.READY_CURLED,
            CalibrationStage.CAPTURING_CURLED,
        ):
            return CalibrationPosition.CURLED
        return None

    @property
    def sample_count(self) -> int:
        """Return retained samples for the endpoint currently shown."""
        position = self.position
        return 0 if position is None else self.collector.sample_count(position)

    def record(self, angle: float | None) -> int:
        """Capture one angle only while an endpoint is actively sampling."""
        positions = {
            CalibrationStage.CAPTURING_EXTENDED: CalibrationPosition.EXTENDED,
            CalibrationStage.CAPTURING_CURLED: CalibrationPosition.CURLED,
        }
        position = positions.get(self.stage)
        if position is None:
            return self.sample_count
        count = self.collector.add(position, angle)
        if self.collector.position_ready(position):
            self.stage = (
                CalibrationStage.READY_CURLED
                if position is CalibrationPosition.EXTENDED
                else CalibrationStage.COMPLETE
            )
        return count

    def build_profile(
        self, calibrated_at: datetime | None = None
    ) -> CalibrationProfile:
        """Return the completed personalized profile."""
        if self.stage is not CalibrationStage.COMPLETE:
            raise CalibrationError("Guided calibration is not complete.")
        return self.collector.build_profile(calibrated_at)

    def reset(self) -> None:
        """Restart the entire guided workflow from extension."""
        self.collector.reset()
        self.stage = CalibrationStage.READY_EXTENDED


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


def profile_from_dict(data: object) -> CalibrationProfile:
    """Parse one stored profile through the same runtime validation."""
    if not isinstance(data, dict):
        raise CalibrationStorageError("Stored calibration profile must be an object.")
    expected = {
        "arm",
        "curled_angle",
        "extended_angle",
        "up_threshold",
        "down_threshold",
        "samples_per_position",
        "calibrated_at",
    }
    if set(data) != expected:
        raise CalibrationStorageError("Stored calibration profile fields are invalid.")
    try:
        numeric_fields = (
            "curled_angle",
            "extended_angle",
            "up_threshold",
            "down_threshold",
        )
        numeric_values = []
        for field in numeric_fields:
            value = data[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field} must be numeric")
            numeric_values.append(float(value))
        samples = data["samples_per_position"]
        if isinstance(samples, bool) or not isinstance(samples, int):
            raise TypeError("samples_per_position must be an integer")
        calibrated_at = data["calibrated_at"]
        if not isinstance(calibrated_at, str):
            raise TypeError("calibrated_at must be text")
        return CalibrationProfile(
            Arm(data["arm"]),
            *numeric_values,
            samples,
            datetime.fromisoformat(calibrated_at),
        )
    except (TypeError, ValueError) as error:
        raise CalibrationStorageError(
            f"Stored calibration profile is invalid: {error}"
        ) from error


class CalibrationStore:
    """Load and save versioned aggregate calibration profiles."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = default_calibration_path() if path is None else path

    def load_all(self) -> dict[Arm, CalibrationProfile]:
        """Return saved profiles, or an empty mapping before first use."""
        if not self.path.exists():
            return {}
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeError) as error:
            raise CalibrationStorageError(
                f"Could not read calibration file {self.path}: {error}"
            ) from error
        if not isinstance(document, dict) or set(document) != {"version", "arms"}:
            raise CalibrationStorageError("Calibration file structure is invalid.")
        if document["version"] != CALIBRATION_SCHEMA_VERSION:
            raise CalibrationStorageError(
                f"Unsupported calibration version: {document['version']}"
            )
        arms = document["arms"]
        if not isinstance(arms, dict):
            raise CalibrationStorageError("Calibration arms must be an object.")
        profiles: dict[Arm, CalibrationProfile] = {}
        for arm_value, stored_profile in arms.items():
            try:
                arm = Arm(arm_value)
            except ValueError as error:
                raise CalibrationStorageError(
                    f"Unsupported calibration arm: {arm_value}"
                ) from error
            parsed = profile_from_dict(stored_profile)
            if parsed.arm is not arm:
                raise CalibrationStorageError(
                    f"Calibration arm key does not match profile: {arm.value}"
                )
            profiles[arm] = parsed
        return profiles

    def save(self, profile: CalibrationProfile) -> Path:
        """Atomically add or replace one arm's aggregate profile."""
        profiles = self.load_all()
        profiles[profile.arm] = profile
        self._write_profiles(profiles)
        return self.path

    def _write_profiles(self, profiles: dict[Arm, CalibrationProfile]) -> None:
        ordered_profiles = sorted(
            profiles.items(), key=lambda item: item[0].value
        )
        document = {
            "version": CALIBRATION_SCHEMA_VERSION,
            "arms": {
                arm.value: profile_to_dict(saved)
                for arm, saved in ordered_profiles
            },
        }
        temporary_path: Path | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                "w",
                encoding="utf-8",
                newline="\n",
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                dir=self.path.parent,
                delete=False,
            ) as temporary:
                json.dump(document, temporary, indent=2, sort_keys=True)
                temporary.write("\n")
                temporary_path = Path(temporary.name)
            os.replace(temporary_path, self.path)
        except OSError as error:
            if temporary_path is not None:
                with suppress(OSError):
                    temporary_path.unlink()
            raise CalibrationStorageError(
                f"Could not save calibration file {self.path}: {error}"
            ) from error

    def load(self, arm: Arm) -> CalibrationProfile | None:
        """Return one arm's saved profile when available."""
        return self.load_all().get(arm)

    def reset(self, arm: Arm) -> bool:
        """Remove one arm profile and preserve any other saved arm."""
        profiles = self.load_all()
        removed = profiles.pop(arm, None) is not None
        if removed:
            self._write_profiles(profiles)
        return removed


def apply_calibration(
    config: AppConfig, profile: CalibrationProfile
) -> AppConfig:
    """Return runtime settings with one matching arm's thresholds applied."""
    if profile.arm is not config.selected_arm:
        raise CalibrationError(
            f"Cannot apply {profile.arm.value} calibration while "
            f"{config.selected_arm.value} arm is selected."
        )
    return replace(
        config,
        up_angle_threshold=profile.up_threshold,
        down_angle_threshold=profile.down_threshold,
    )


def load_calibrated_config(
    config: AppConfig, store: CalibrationStore | None = None
) -> tuple[AppConfig, CalibrationProfile | None]:
    """Apply a saved selected-arm profile, preserving defaults when absent."""
    selected_store = CalibrationStore() if store is None else store
    profile = selected_store.load(config.selected_arm)
    if profile is None:
        return config, None
    return apply_calibration(config, profile), profile


def format_calibration_profile(profile: CalibrationProfile) -> str:
    """Format one saved profile for a concise command-line report."""
    return (
        f"arm={profile.arm.value}, curled={profile.curled_angle:.1f}, "
        f"extended={profile.extended_angle:.1f}, "
        f"up={profile.up_threshold:.1f}, down={profile.down_threshold:.1f}, "
        f"samples={profile.samples_per_position}, "
        f"calibrated={profile.calibrated_at.isoformat()}"
    )
