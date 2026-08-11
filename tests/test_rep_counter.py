from math import cos, radians, sin

import pytest

from repvision.config import AppConfig, Arm
from repvision.pose_detector import ArmLandmarks, Landmark, Point2D
from repvision.rep_counter import (
    CurlTracker,
    CurlUpdate,
    MovementStage,
    RepCounter,
    RepUpdate,
)


def test_movement_stage_values_are_overlay_friendly() -> None:
    assert MovementStage.UNKNOWN.value == "unknown"
    assert MovementStage.DOWN.value == "down"
    assert MovementStage.UP.value == "up"


def counter(**overrides: float | int) -> RepCounter:
    settings: dict[str, float | int] = {
        "up_threshold": 50.0,
        "down_threshold": 155.0,
        "confirmation_frames": 3,
        "cooldown_seconds": 0.3,
    }
    settings.update(overrides)
    return RepCounter(**settings)  # type: ignore[arg-type]


def arm_at_angle(angle: float, confidence: float = 0.9) -> ArmLandmarks:
    elbow = Landmark(Point2D(0.0, 0.0), confidence)
    shoulder = Landmark(Point2D(1.0, 0.0), confidence)
    angle_radians = radians(angle)
    wrist = Landmark(Point2D(cos(angle_radians), sin(angle_radians)), confidence)
    hip = Landmark(Point2D(1.0, 1.0), confidence)
    return ArmLandmarks(Arm.RIGHT, shoulder, elbow, wrist, hip)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"up_threshold": -1.0}, "thresholds"),
        ({"down_threshold": 181.0}, "thresholds"),
        ({"up_threshold": 155.0, "down_threshold": 50.0}, "thresholds"),
        ({"confirmation_frames": 0}, "confirmation_frames"),
        ({"cooldown_seconds": -0.1}, "cooldown_seconds"),
    ],
)
def test_counter_rejects_invalid_configuration(
    overrides: dict[str, float | int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        counter(**overrides)


def test_counter_uses_shared_application_configuration() -> None:
    config = AppConfig(
        up_angle_threshold=45.0,
        down_angle_threshold=160.0,
        confirmation_frames=4,
        cooldown_seconds=0.75,
    )

    tracker = RepCounter.from_config(config)

    assert tracker.up_threshold == 45.0
    assert tracker.down_threshold == 160.0
    assert tracker.confirmation_frames == 4
    assert tracker.cooldown_seconds == 0.75


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (180.0, MovementStage.DOWN),
        (155.0, MovementStage.DOWN),
        (154.9, None),
        (90.0, None),
        (50.1, None),
        (50.0, MovementStage.UP),
        (0.0, MovementStage.UP),
        (None, None),
        (float("nan"), None),
    ],
)
def test_angle_classification_uses_threshold_boundaries(
    angle: float | None, expected: MovementStage | None
) -> None:
    assert counter().classify(angle) is expected


def test_new_counter_snapshot_is_unknown_and_empty() -> None:
    assert counter().snapshot() == RepUpdate(0, MovementStage.UNKNOWN)


def test_curl_update_keeps_measurement_and_counter_state_together() -> None:
    update = CurlUpdate(160.0, 158.0, 2, MovementStage.DOWN)

    assert update.raw_angle == 160.0
    assert update.smoothed_angle == 158.0
    assert update.count == 2
    assert update.stage is MovementStage.DOWN
    assert not update.rep_completed


def test_endpoint_requires_configured_confirmation_frames() -> None:
    tracker = counter(confirmation_frames=3)

    first = tracker.update(160.0, timestamp=0.0)
    second = tracker.update(160.0, timestamp=0.1)
    third = tracker.update(160.0, timestamp=0.2)

    assert first.stage is MovementStage.UNKNOWN
    assert second.stage is MovementStage.UNKNOWN
    assert not second.transition_accepted
    assert third.stage is MovementStage.DOWN
    assert third.transition_accepted


def test_confirmed_down_to_up_transition_counts_one_rep() -> None:
    tracker = counter(confirmation_frames=2)

    tracker.update(160.0, timestamp=0.0)
    down = tracker.update(160.0, timestamp=0.1)
    tracker.update(40.0, timestamp=1.0)
    completed = tracker.update(40.0, timestamp=1.1)

    assert down.stage is MovementStage.DOWN
    assert down.count == 0
    assert completed == RepUpdate(
        count=1,
        stage=MovementStage.UP,
        transition_accepted=True,
        rep_completed=True,
        rep_duration_seconds=1.0,
    )


def test_repeated_frames_in_same_state_do_not_double_count() -> None:
    tracker = counter(confirmation_frames=2)
    for timestamp, angle in enumerate([160.0, 160.0, 40.0, 40.0]):
        tracker.update(angle, timestamp=float(timestamp))

    repeated_updates = [
        tracker.update(40.0, timestamp=float(timestamp)) for timestamp in range(4, 10)
    ]

    assert tracker.count == 1
    assert all(update.count == 1 for update in repeated_updates)
    assert all(not update.rep_completed for update in repeated_updates)


def test_middle_range_jitter_does_not_change_stage() -> None:
    tracker = counter(confirmation_frames=2)
    tracker.update(160.0, timestamp=0.0)
    tracker.update(160.0, timestamp=0.1)

    for timestamp, angle in enumerate([120.0, 80.0, 130.0, 55.0], start=1):
        update = tracker.update(angle, timestamp=float(timestamp))
        assert update.stage is MovementStage.DOWN
        assert update.count == 0
        assert not update.transition_accepted


def test_middle_angle_breaks_pending_confirmation_sequence() -> None:
    tracker = counter(confirmation_frames=2)
    tracker.update(160.0, timestamp=0.0)
    tracker.update(160.0, timestamp=0.1)
    tracker.update(45.0, timestamp=1.0)
    tracker.update(90.0, timestamp=1.1)

    still_down = tracker.update(45.0, timestamp=1.2)

    assert still_down.stage is MovementStage.DOWN
    assert still_down.count == 0


@pytest.mark.parametrize(
    "angles",
    [
        [90.0, 80.0, 70.0, 60.0],
        [160.0, 160.0, 100.0, 80.0, 60.0],
        [40.0, 40.0, 90.0, 100.0],
    ],
)
def test_partial_movements_do_not_count(angles: list[float]) -> None:
    tracker = counter(confirmation_frames=2)

    for timestamp, angle in enumerate(angles):
        tracker.update(angle, timestamp=float(timestamp))

    assert tracker.count == 0


def test_starting_in_up_position_does_not_create_a_rep() -> None:
    tracker = counter(confirmation_frames=2)

    tracker.update(40.0, timestamp=0.0)
    update = tracker.update(40.0, timestamp=0.1)

    assert update.stage is MovementStage.UP
    assert update.count == 0
    assert not update.rep_completed


@pytest.mark.parametrize("missing", [None, float("nan"), float("inf")])
def test_missing_measurement_breaks_pending_transition(
    missing: float | None,
) -> None:
    tracker = counter(confirmation_frames=2)
    tracker.update(160.0, timestamp=0.0)
    tracker.update(160.0, timestamp=0.1)
    tracker.update(40.0, timestamp=1.0)

    missing_update = tracker.update(missing, timestamp=1.1)
    first_visible = tracker.update(40.0, timestamp=1.2)
    completed = tracker.update(40.0, timestamp=1.3)

    assert missing_update.stage is MovementStage.DOWN
    assert missing_update.count == 0
    assert first_visible.count == 0
    assert completed.count == 1


def test_cooldown_delays_implausibly_fast_second_rep() -> None:
    tracker = counter(confirmation_frames=1, cooldown_seconds=0.5)
    tracker.update(160.0, timestamp=0.0)
    first_rep = tracker.update(40.0, timestamp=1.0)
    tracker.update(160.0, timestamp=1.1)

    too_soon = tracker.update(40.0, timestamp=1.2)
    after_cooldown = tracker.update(40.0, timestamp=1.5)

    assert first_rep.count == 1
    assert too_soon.count == 1
    assert too_soon.stage is MovementStage.DOWN
    assert not too_soon.rep_completed
    assert after_cooldown.count == 2
    assert after_cooldown.stage is MovementStage.UP
    assert after_cooldown.rep_completed


def test_counter_reset_clears_all_movement_state() -> None:
    tracker = counter(confirmation_frames=1)
    tracker.update(160.0, timestamp=0.0)
    tracker.update(40.0, timestamp=1.0)
    tracker.update(160.0, timestamp=1.1)

    tracker.reset()

    assert tracker.snapshot() == RepUpdate(0, MovementStage.UNKNOWN)
    after_reset = tracker.update(40.0, timestamp=1.2)
    assert after_reset.stage is MovementStage.UP
    assert after_reset.count == 0


def test_curl_tracker_counts_reliable_smoothed_arm_movements() -> None:
    tracker = CurlTracker(
        AppConfig(
            smoothing_window=1,
            confirmation_frames=2,
            cooldown_seconds=0.0,
        )
    )

    tracker.update(arm_at_angle(160.0), timestamp=0.0)
    tracker.update(arm_at_angle(160.0), timestamp=0.1)
    tracker.update(arm_at_angle(40.0), timestamp=1.0)
    completed = tracker.update(arm_at_angle(40.0), timestamp=1.1)

    assert completed.raw_angle == pytest.approx(40.0)
    assert completed.smoothed_angle == pytest.approx(40.0)
    assert completed.count == 1
    assert completed.stage is MovementStage.UP
    assert completed.rep_completed


def test_curl_tracker_does_not_advance_on_low_confidence_landmarks() -> None:
    tracker = CurlTracker(
        AppConfig(
            confidence_threshold=0.5,
            smoothing_window=1,
            confirmation_frames=2,
            cooldown_seconds=0.0,
        )
    )
    tracker.update(arm_at_angle(160.0), timestamp=0.0)

    unreliable = tracker.update(
        arm_at_angle(160.0, confidence=0.4), timestamp=0.1
    )
    first_visible = tracker.update(arm_at_angle(160.0), timestamp=0.2)
    confirmed = tracker.update(arm_at_angle(160.0), timestamp=0.3)

    assert unreliable.raw_angle is None
    assert unreliable.smoothed_angle is None
    assert unreliable.stage is MovementStage.UNKNOWN
    assert tracker.smoother.sample_count == 1
    assert first_visible.stage is MovementStage.UNKNOWN
    assert confirmed.stage is MovementStage.DOWN


def test_curl_tracker_reset_clears_smoothing_and_counter() -> None:
    tracker = CurlTracker(
        AppConfig(
            smoothing_window=1,
            confirmation_frames=1,
            cooldown_seconds=0.0,
        )
    )
    tracker.update(arm_at_angle(160.0), timestamp=0.0)
    tracker.update(arm_at_angle(40.0), timestamp=1.0)

    tracker.reset()

    assert tracker.smoother.value is None
    assert tracker.counter.snapshot() == RepUpdate(0, MovementStage.UNKNOWN)


def test_counter_reports_confirmed_rep_duration() -> None:
    tracker = counter(confirmation_frames=1)
    tracker.update(160.0, timestamp=10.0)

    completed = tracker.update(40.0, timestamp=11.25)

    assert completed.rep_duration_seconds == pytest.approx(1.25)
