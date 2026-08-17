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
    assert config.upper_arm_drift_threshold == 30.0
    assert config.input_size == 640
    assert config.output_directory == Path("outputs")
    assert config.calibration_sample_target == 20
    assert config.calibration_minimum_range == 60.0
    assert config.calibration_threshold_margin == 10.0
    assert not config.audio_cues


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


@pytest.mark.parametrize(
    ("up", "down"),
    [(-1.0, 155.0), (50.0, 181.0), (155.0, 50.0), (50.0, 50.0)],
)
def test_angle_thresholds_must_define_an_ordered_range(
    up: float, down: float
) -> None:
    with pytest.raises(
        ValueError, match="angle thresholds must satisfy 0 <= up < down <= 180"
    ):
        AppConfig(up_angle_threshold=up, down_angle_threshold=down)


@pytest.mark.parametrize(
    ("keyword", "message"),
    [
        ("confirmation_frames", "confirmation_frames must be positive"),
        ("smoothing_window", "smoothing_window must be positive"),
    ],
)
def test_frame_windows_must_be_positive(keyword: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        AppConfig(**{keyword: 0})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("cooldown_seconds", -0.1, "cooldown_seconds must not be negative"),
        ("input_size", 0, "input_size must be positive"),
    ],
)
def test_performance_settings_reject_invalid_values(
    keyword: str, value: int | float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        AppConfig(**{keyword: value})  # type: ignore[arg-type]


@pytest.mark.parametrize("threshold", [0.0, -1.0, 90.1])
def test_arm_drift_threshold_stays_conservative(threshold: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 90"):
        AppConfig(upper_arm_drift_threshold=threshold)


@pytest.mark.parametrize("sample_target", [0, 1, 2])
def test_calibration_requires_enough_samples(sample_target: int) -> None:
    with pytest.raises(ValueError, match="calibration_sample_target"):
        AppConfig(calibration_sample_target=sample_target)


@pytest.mark.parametrize("minimum_range", [0.0, -1.0, 180.0, 181.0])
def test_calibration_range_stays_physical(minimum_range: float) -> None:
    with pytest.raises(ValueError, match="calibration_minimum_range"):
        AppConfig(calibration_minimum_range=minimum_range)


@pytest.mark.parametrize("margin", [0.0, -1.0, 30.0, 31.0])
def test_calibration_margin_preserves_threshold_order(margin: float) -> None:
    with pytest.raises(ValueError, match="calibration_threshold_margin"):
        AppConfig(calibration_threshold_margin=margin)


def test_audio_cues_require_boolean_setting() -> None:
    with pytest.raises(ValueError, match="audio_cues"):
        AppConfig(audio_cues=1)  # type: ignore[arg-type]
