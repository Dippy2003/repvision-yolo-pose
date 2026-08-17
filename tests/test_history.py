from datetime import datetime

import pytest

from repvision.config import Arm
from repvision.history import SessionHistory, format_session_history
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


def test_format_session_history_shows_totals_and_recent_workouts() -> None:
    first = summary(8, 60.0, 1)
    second = SessionSummary(
        datetime(2026, 8, 18),
        "bicep_curl",
        Arm.LEFT,
        90.0,
        12,
        2,
        None,
    )

    report = format_session_history((first, second), recent_limit=1)

    assert "Sessions: 2" in report
    assert "Total time: 2.5 minutes" in report
    assert "Total reps: 20" in report
    assert "Best session: 12 reps" in report
    assert "2026-08-18T00:00 | left | 12 reps" in report
    assert "2026-08-17" not in report


def test_format_session_history_handles_empty_log() -> None:
    assert format_session_history(()) == "No aggregate workout sessions found."


def test_format_session_history_requires_positive_limit() -> None:
    with pytest.raises(ValueError, match="recent_limit"):
        format_session_history((), recent_limit=0)
