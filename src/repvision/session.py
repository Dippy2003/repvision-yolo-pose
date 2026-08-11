"""Aggregate workout session statistics and CSV persistence."""

from dataclasses import dataclass
from datetime import datetime

from repvision.config import Arm


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
