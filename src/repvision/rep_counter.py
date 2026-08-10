"""Confirmed-frame bicep-curl movement state and repetition counting."""

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite


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
