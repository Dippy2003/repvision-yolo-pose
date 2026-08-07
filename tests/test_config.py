from pathlib import Path

import pytest

from repvision.config import AppConfig, Arm


def test_arm_values_are_command_line_friendly() -> None:
    assert Arm.LEFT.value == "left"
    assert Arm.RIGHT.value == "right"


def test_application_defaults_match_initial_tracking_setup() -> None:
    config = AppConfig()

    assert config.model_name == "yolo26n-pose.pt"
    assert config.camera_index == 0
    assert config.selected_arm is Arm.RIGHT
    assert config.confidence_threshold == 0.5
    assert config.up_angle_threshold == 50.0
    assert config.down_angle_threshold == 155.0
    assert config.confirmation_frames == 3
    assert config.smoothing_window == 5
    assert config.cooldown_seconds == 0.3
    assert config.input_size == 640
    assert config.output_directory == Path("outputs")


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("model_name", "  ", "model_name must not be empty"),
        ("camera_index", -1, "camera_index must be zero or greater"),
    ],
)
def test_invalid_capture_source_settings_are_rejected(
    keyword: str, value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        AppConfig(**{keyword: value})  # type: ignore[arg-type]


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_confidence_threshold_must_be_a_probability(threshold: float) -> None:
    with pytest.raises(
        ValueError, match="confidence_threshold must be between 0 and 1"
    ):
        AppConfig(confidence_threshold=threshold)
