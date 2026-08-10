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
