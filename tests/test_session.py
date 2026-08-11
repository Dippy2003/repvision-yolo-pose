from datetime import datetime

from repvision.config import Arm
from repvision.form_checker import FeedbackMessage, FormFeedback
from repvision.rep_counter import CurlUpdate, MovementStage
from repvision.session import SESSION_HEADERS, SessionAccumulator, SessionSummary


def test_session_summary_formats_stable_aggregate_row() -> None:
    summary = SessionSummary(
        datetime(2026, 8, 11, 9, 30, 5),
        "bicep_curl",
        Arm.LEFT,
        65.127,
        8,
        2,
        2.345,
    )

    assert summary.as_csv_row() == (
        "2026-08-11T09:30:05",
        "bicep_curl",
        "left",
        "65.13",
        "8",
        "2",
        "2.35",
    )
    assert len(summary.as_csv_row()) == len(SESSION_HEADERS)


def test_session_summary_leaves_unavailable_rep_average_empty() -> None:
    summary = SessionSummary(
        datetime(2026, 8, 11), "bicep_curl", Arm.RIGHT, 1.0, 0, 0, None
    )

    assert summary.as_csv_row()[-1] == ""


def test_session_accumulator_tracks_reps_warnings_and_average_duration() -> None:
    accumulator = SessionAccumulator(datetime(2026, 8, 11, 10), 100.0)
    warning = FormFeedback(FeedbackMessage.ELBOW_DRIFT, is_form_warning=True)
    accumulator.record(
        CurlUpdate(None, 40.0, 1, MovementStage.UP, True, 1.5), warning
    )
    accumulator.record(
        CurlUpdate(None, 160.0, 2, MovementStage.DOWN, True, 2.5), warning
    )

    summary = accumulator.summary(Arm.RIGHT, 110.0)

    assert summary.duration_seconds == 10.0
    assert summary.repetitions == 2
    assert summary.warning_count == 1
    assert summary.average_rep_duration_seconds == 2.0


def test_session_accumulator_reset_starts_fresh_statistics() -> None:
    accumulator = SessionAccumulator(datetime(2026, 8, 11, 10), 100.0)
    accumulator.record(
        CurlUpdate(None, 40.0, 1, MovementStage.UP, True, 1.5),
        FormFeedback(FeedbackMessage.ELBOW_DRIFT, is_form_warning=True),
    )

    accumulator.reset(datetime(2026, 8, 11, 11), 200.0)
    summary = accumulator.summary(Arm.LEFT, 205.0)

    assert summary.started_at == datetime(2026, 8, 11, 11)
    assert summary.duration_seconds == 5.0
    assert summary.repetitions == 0
    assert summary.warning_count == 0
    assert summary.average_rep_duration_seconds is None
