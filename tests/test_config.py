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
