"""Aggregate workout session statistics and CSV persistence."""

import csv
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from pathlib import Path
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


class SessionLogError(RuntimeError):
    """Raised when aggregate session values cannot be persisted."""


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

    def __post_init__(self) -> None:
        if not self.exercise.strip():
            raise ValueError("exercise must not be empty")
        if not isfinite(self.duration_seconds) or self.duration_seconds < 0.0:
            raise ValueError("duration_seconds must be finite and non-negative")
        if self.repetitions < 0:
            raise ValueError("repetitions must be non-negative")
        if self.warning_count < 0:
            raise ValueError("warning_count must be non-negative")
        average = self.average_rep_duration_seconds
        if average is not None and (not isfinite(average) or average < 0.0):
            raise ValueError(
                "average_rep_duration_seconds must be finite and non-negative"
            )

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

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> "SessionSummary":
        """Parse one aggregate CSV row through normal value validation."""
        if set(row) != set(SESSION_HEADERS):
            raise ValueError("session row fields do not match the expected schema")
        average_text = row["average_rep_duration_seconds"]
        return cls(
            datetime.fromisoformat(row["datetime"]),
            row["exercise"],
            Arm(row["arm"]),
            float(row["duration_seconds"]),
            int(row["repetitions"]),
            int(row["warning_count"]),
            None if average_text == "" else float(average_text),
        )


class SessionAccumulator:
    """Collect aggregate-only statistics while frames remain in memory."""

    def __init__(self, started_at: datetime, started_monotonic: float) -> None:
        self._validate_start_time(started_monotonic)
        self.started_at = started_at
        self.started_monotonic = started_monotonic
        self.repetitions = 0
        self.warning_counter = WarningCounter()
        self._rep_durations: list[float] = []

    def record(self, update: CurlUpdate, feedback: FormFeedback) -> None:
        """Consume one processed frame without retaining image data."""
        self.repetitions = update.count
        self.warning_counter.update(feedback)
        duration = update.rep_duration_seconds
        if (
            update.rep_completed
            and duration is not None
            and isfinite(duration)
            and duration >= 0.0
        ):
            self._rep_durations.append(duration)

    def summary(self, arm: Arm, ended_monotonic: float) -> SessionSummary:
        """Build a bicep-curl summary at a supplied monotonic end time."""
        if not isfinite(ended_monotonic):
            raise ValueError("ended_monotonic must be finite")
        if ended_monotonic < self.started_monotonic:
            raise ValueError("ended_monotonic must not precede the session start")
        average = fmean(self._rep_durations) if self._rep_durations else None
        return SessionSummary(
            self.started_at,
            "bicep_curl",
            arm,
            ended_monotonic - self.started_monotonic,
            self.repetitions,
            self.warning_counter.count,
            average,
        )

    def reset(self, started_at: datetime, started_monotonic: float) -> None:
        """Begin fresh aggregate statistics after an explicit workout reset."""
        self._validate_start_time(started_monotonic)
        self.started_at = started_at
        self.started_monotonic = started_monotonic
        self.repetitions = 0
        self.warning_counter.reset()
        self._rep_durations.clear()

    @staticmethod
    def _validate_start_time(started_monotonic: float) -> None:
        if not isfinite(started_monotonic):
            raise ValueError("started_monotonic must be finite")


class SessionLogger:
    """Append aggregate session summaries to one local CSV file."""

    def __init__(self, output_directory: Path) -> None:
        self.output_directory = output_directory

    @property
    def path(self) -> Path:
        """Return the configured aggregate log path."""
        return self.output_directory / "sessions.csv"

    def save(self, summary: SessionSummary) -> Path:
        """Create or append to the aggregate CSV and return its path."""
        try:
            self.output_directory.mkdir(parents=True, exist_ok=True)
            include_header = self._include_header()
            with self.path.open("a", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                if include_header:
                    writer.writerow(SESSION_HEADERS)
                writer.writerow(summary.as_csv_row())
        except (OSError, UnicodeError) as error:
            raise SessionLogError(
                f"Could not save aggregate session log: {error}"
            ) from error
        return self.path

    def _include_header(self) -> bool:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return True
        with self.path.open(encoding="utf-8", newline="") as file:
            existing_header = tuple(next(csv.reader(file), ()))
        if existing_header != SESSION_HEADERS:
            raise SessionLogError(
                f"Existing session log has an incompatible header: {self.path}"
            )
        return False
