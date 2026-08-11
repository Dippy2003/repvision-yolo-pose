"""Aggregate workout session statistics and CSV persistence."""

from dataclasses import dataclass
from datetime import datetime
from statistics import fmean

from repvision.config import Arm
from repvision.form_checker import FormFeedback, WarningCounter
from repvision.rep_counter import CurlUpdate

SESSION_HEADERS = (
    "datetime",
    "exercise",
    "arm",
    "duration_seconds",
    "repetitions",
    "warning_count",
    "average_rep_duration_seconds",
)


@dataclass(frozen=True, slots=True)
class SessionSummary:
    """Privacy-safe aggregate values saved after a workout."""

    started_at: datetime
    exercise: str
    arm: Arm
    duration_seconds: float
    repetitions: int
    warning_count: int
    average_rep_duration_seconds: float | None

    def as_csv_row(self) -> tuple[str, ...]:
        """Format aggregate values into the stable CSV schema."""
        average = self.average_rep_duration_seconds
        return (
            self.started_at.isoformat(timespec="seconds"),
            self.exercise,
            self.arm.value,
            f"{self.duration_seconds:.2f}",
            str(self.repetitions),
            str(self.warning_count),
            "" if average is None else f"{average:.2f}",
        )


class SessionAccumulator:
    """Collect aggregate-only statistics while frames remain in memory."""

    def __init__(self, started_at: datetime, started_monotonic: float) -> None:
        self.started_at = started_at
        self.started_monotonic = started_monotonic
        self.repetitions = 0
        self.warning_counter = WarningCounter()
        self._rep_durations: list[float] = []

    def record(self, update: CurlUpdate, feedback: FormFeedback) -> None:
        """Consume one processed frame without retaining image data."""
        self.repetitions = update.count
        self.warning_counter.update(feedback)
        if update.rep_completed and update.rep_duration_seconds is not None:
            self._rep_durations.append(update.rep_duration_seconds)

    def summary(self, arm: Arm, ended_monotonic: float) -> SessionSummary:
        """Build a bicep-curl summary at a supplied monotonic end time."""
        average = fmean(self._rep_durations) if self._rep_durations else None
        return SessionSummary(
            self.started_at,
            "bicep_curl",
            arm,
            max(0.0, ended_monotonic - self.started_monotonic),
            self.repetitions,
            self.warning_counter.count,
            average,
        )

    def reset(self, started_at: datetime, started_monotonic: float) -> None:
        """Begin fresh aggregate statistics after an explicit workout reset."""
        self.started_at = started_at
        self.started_monotonic = started_monotonic
        self.repetitions = 0
        self.warning_counter.reset()
        self._rep_durations.clear()
