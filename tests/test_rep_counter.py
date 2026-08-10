from repvision.rep_counter import MovementStage


def test_movement_stage_values_are_overlay_friendly() -> None:
    assert MovementStage.UNKNOWN.value == "unknown"
    assert MovementStage.DOWN.value == "down"
    assert MovementStage.UP.value == "up"
