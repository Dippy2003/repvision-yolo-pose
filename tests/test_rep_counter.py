import pytest

from repvision.rep_counter import MovementStage, RepCounter


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
