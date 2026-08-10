import pytest

from repvision.rep_counter import MovementStage, RepCounter, RepUpdate


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
