from datetime import datetime

from repvision.config import Arm
from repvision.history import SessionHistory
from repvision.session import SessionSummary


def summary(repetitions: int, duration: float, warnings: int) -> SessionSummary:
    return SessionSummary(
        datetime(2026, 8, 17),
        "bicep_curl",
        Arm.RIGHT,
        duration,
        repetitions,
        warnings,
        None,
    )


def test_session_history_aggregates_privacy_safe_totals() -> None:
    history = SessionHistory.from_summaries(
        (summary(8, 60.0, 1), summary(12, 90.0, 2))
    )

    assert history.sessions == 2
    assert history.total_duration_seconds == 150.0
    assert history.total_repetitions == 20
    assert history.total_warnings == 3
    assert history.best_repetitions == 12
    assert history.average_repetitions == 10.0


def test_session_history_handles_no_saved_workouts() -> None:
    history = SessionHistory.from_summaries(())

    assert history.sessions == 0
    assert history.total_duration_seconds == 0.0
    assert history.total_repetitions == 0
    assert history.best_repetitions == 0
    assert history.average_repetitions == 0.0
