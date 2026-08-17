"""Privacy-safe aggregate workout history reporting."""

from dataclasses import dataclass
from statistics import fmean

from repvision.session import SessionSummary


@dataclass(frozen=True, slots=True)
class SessionHistory:
    """Aggregate trends derived without frame or keypoint data."""

    sessions: int
    total_duration_seconds: float
    total_repetitions: int
    total_warnings: int
    best_repetitions: int
    average_repetitions: float

    @classmethod
    def from_summaries(
        cls, summaries: tuple[SessionSummary, ...]
    ) -> "SessionHistory":
        """Calculate safe totals from validated session summaries."""
        repetitions = [summary.repetitions for summary in summaries]
        return cls(
            sessions=len(summaries),
            total_duration_seconds=sum(
                summary.duration_seconds for summary in summaries
            ),
            total_repetitions=sum(repetitions),
            total_warnings=sum(summary.warning_count for summary in summaries),
            best_repetitions=max(repetitions, default=0),
            average_repetitions=fmean(repetitions) if repetitions else 0.0,
        )
