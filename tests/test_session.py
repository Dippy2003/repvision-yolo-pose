from datetime import datetime

from repvision.config import Arm
from repvision.session import SESSION_HEADERS, SessionSummary


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
