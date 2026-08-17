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


def format_session_history(
    summaries: tuple[SessionSummary, ...], *, recent_limit: int = 5
) -> str:
    """Format totals and the newest aggregate workouts for the terminal."""
    if recent_limit < 1:
        raise ValueError("recent_limit must be positive")
    if not summaries:
        return "No aggregate workout sessions found."
    history = SessionHistory.from_summaries(summaries)
    lines = [
        "RepVision session history",
        f"Sessions: {history.sessions}",
        f"Total time: {history.total_duration_seconds / 60.0:.1f} minutes",
        f"Total reps: {history.total_repetitions}",
        f"Average reps: {history.average_repetitions:.1f}",
        f"Best session: {history.best_repetitions} reps",
        f"Form warnings: {history.total_warnings}",
        "Recent sessions:",
    ]
    newest = sorted(summaries, key=lambda item: item.started_at, reverse=True)
    for summary in newest[:recent_limit]:
        lines.append(
            f"- {summary.started_at.isoformat(timespec='minutes')} | "
            f"{summary.arm.value} | {summary.repetitions} reps | "
            f"{summary.duration_seconds:.1f}s"
        )
    return "\n".join(lines)
