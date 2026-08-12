from datetime import datetime

import pytest

from repvision.config import Arm
from repvision.form_checker import FeedbackMessage, FormFeedback
from repvision.rep_counter import CurlUpdate, MovementStage
from repvision.session import (
    SESSION_HEADERS,
    SessionAccumulator,
    SessionLogger,
    SessionSummary,
)


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


def test_session_logger_creates_aggregate_csv(tmp_path) -> None:
    summary = SessionSummary(
        datetime(2026, 8, 11, 12), "bicep_curl", Arm.RIGHT, 12.0, 3, 1, 2.0
    )

    saved_path = SessionLogger(tmp_path / "outputs").save(summary)

    assert saved_path == tmp_path / "outputs" / "sessions.csv"
    lines = saved_path.read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",") == list(SESSION_HEADERS)
    assert lines[1] == "2026-08-11T12:00:00,bicep_curl,right,12.00,3,1,2.00"


def test_session_logger_appends_without_repeating_header(tmp_path) -> None:
    logger = SessionLogger(tmp_path)
    first = SessionSummary(
        datetime(2026, 8, 11, 12), "bicep_curl", Arm.RIGHT, 10.0, 2, 0, 1.0
    )
    second = SessionSummary(
        datetime(2026, 8, 11, 13), "bicep_curl", Arm.LEFT, 20.0, 4, 1, 2.0
    )

    logger.save(first)
    logger.save(second)

    lines = logger.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert lines.count(",".join(SESSION_HEADERS)) == 1
    assert ",left,20.00,4,1,2.00" in lines[2]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("exercise", " "),
        ("duration_seconds", -1.0),
        ("duration_seconds", float("nan")),
        ("repetitions", -1),
        ("warning_count", -1),
        ("average_rep_duration_seconds", -0.1),
        ("average_rep_duration_seconds", float("inf")),
    ],
)
def test_session_summary_rejects_invalid_aggregate_values(
    field: str, value: object
) -> None:
    values: dict[str, object] = {
        "started_at": datetime(2026, 8, 12),
        "exercise": "bicep_curl",
        "arm": Arm.RIGHT,
        "duration_seconds": 1.0,
        "repetitions": 0,
        "warning_count": 0,
        "average_rep_duration_seconds": None,
    }
    values[field] = value

    with pytest.raises(ValueError):
        SessionSummary(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf")])
def test_session_accumulator_rejects_invalid_start_time(timestamp: float) -> None:
    with pytest.raises(ValueError, match="started_monotonic"):
        SessionAccumulator(datetime(2026, 8, 12), timestamp)


def test_session_summary_rejects_end_before_start() -> None:
    accumulator = SessionAccumulator(datetime(2026, 8, 12), 10.0)

    with pytest.raises(ValueError, match="must not precede"):
        accumulator.summary(Arm.RIGHT, 9.0)
