"""Confirmed-frame bicep-curl movement state and repetition counting."""

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from time import monotonic

from repvision.angles import AngleSmoother, calculate_arm_angle
from repvision.config import AppConfig
from repvision.pose_detector import ArmLandmarks


class MovementStage(StrEnum):
    """Accepted position of the selected arm."""

    UNKNOWN = "unknown"
    DOWN = "down"
    UP = "up"


@dataclass(frozen=True, slots=True)
class RepUpdate:
    """Observable result after processing one angle measurement."""

    count: int
    stage: MovementStage
    transition_accepted: bool = False
    rep_completed: bool = False


@dataclass(frozen=True, slots=True)
class CurlUpdate:
    """Combined measurement, smoothing, and repetition result for one frame."""

    raw_angle: float | None
    smoothed_angle: float | None
    count: int
    stage: MovementStage
    rep_completed: bool = False


class RepCounter:
    """Count confirmed full-range down-to-up curl movements."""

    def __init__(
        self,
        *,
        up_threshold: float,
        down_threshold: float,
        confirmation_frames: int,
        cooldown_seconds: float = 0.0,
    ) -> None:
        if not 0.0 <= up_threshold < down_threshold <= 180.0:
            raise ValueError("thresholds must satisfy 0 <= up < down <= 180")
        if confirmation_frames < 1:
            raise ValueError("confirmation_frames must be positive")
        if cooldown_seconds < 0.0:
            raise ValueError("cooldown_seconds must not be negative")
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold
        self.confirmation_frames = confirmation_frames
        self.cooldown_seconds = cooldown_seconds
        self.count = 0
        self.stage = MovementStage.UNKNOWN
        self._candidate: MovementStage | None = None
        self._candidate_frames = 0
        self._last_rep_time: float | None = None

    @classmethod
    def from_config(cls, config: AppConfig) -> "RepCounter":
        """Build a counter from the shared application configuration."""
        return cls(
            up_threshold=config.up_angle_threshold,
            down_threshold=config.down_angle_threshold,
            confirmation_frames=config.confirmation_frames,
            cooldown_seconds=config.cooldown_seconds,
        )

    def snapshot(
        self, *, transition_accepted: bool = False, rep_completed: bool = False
    ) -> RepUpdate:
        """Return the current public counter state."""
        return RepUpdate(self.count, self.stage, transition_accepted, rep_completed)

    def classify(self, angle: float | None) -> MovementStage | None:
        """Classify an angle only when it crosses a configured endpoint."""
        if angle is None or not isfinite(angle):
            return None
        if not 0.0 <= angle <= 180.0:
            raise ValueError("angle must be between 0 and 180 degrees")
        if angle >= self.down_threshold:
            return MovementStage.DOWN
        if angle <= self.up_threshold:
            return MovementStage.UP
        return None

    def update(
        self, angle: float | None, *, timestamp: float | None = None
    ) -> RepUpdate:
        """Process one angle without accepting unconfirmed endpoint changes."""
        target = self.classify(angle)
        if target is None or target is self.stage:
            self._clear_candidate()
            return self.snapshot()

        if target is self._candidate:
            self._candidate_frames += 1
        else:
            self._candidate = target
            self._candidate_frames = 1
        if self._candidate_frames < self.confirmation_frames:
            return self.snapshot()

        now = monotonic() if timestamp is None else timestamp
        rep_completed = self.stage is MovementStage.DOWN and target is MovementStage.UP
        if (
            rep_completed
            and self._last_rep_time is not None
            and now - self._last_rep_time < self.cooldown_seconds
        ):
            return self.snapshot()
        if rep_completed:
            self.count += 1
            self._last_rep_time = now
        self.stage = target
        self._clear_candidate()
        return self.snapshot(transition_accepted=True, rep_completed=rep_completed)

    def _clear_candidate(self) -> None:
        self._candidate = None
        self._candidate_frames = 0

    def reset(self) -> None:
        """Clear repetitions, accepted stage, pending transition, and cooldown."""
        self.count = 0
        self.stage = MovementStage.UNKNOWN
        self._last_rep_time = None
        self._clear_candidate()


class CurlTracker:
    """Connect reliable arm landmarks to smoothing and repetition state."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.smoother = AngleSmoother(config.smoothing_window)
        self.counter = RepCounter.from_config(config)

    def update(
        self,
        landmarks: ArmLandmarks | None,
        *,
        timestamp: float | None = None,
    ) -> CurlUpdate:
        """Process one selected-arm observation."""
        raw_angle = calculate_arm_angle(landmarks, self.config.confidence_threshold)
        smoothed_angle = self.smoother.add(raw_angle)
        rep_update = self.counter.update(smoothed_angle, timestamp=timestamp)
        return CurlUpdate(
            raw_angle,
            smoothed_angle,
            rep_update.count,
            rep_update.stage,
            rep_update.rep_completed,
        )
