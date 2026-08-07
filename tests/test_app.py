from repvision.app import build_parser, config_from_args, main
from repvision.config import Arm


def test_cli_overrides_foundation_settings() -> None:
    args = build_parser().parse_args(
        ["--model", "custom-pose.pt", "--camera-index", "2", "--arm", "left"]
    )

    config = config_from_args(args)

    assert config.model_name == "custom-pose.pt"
    assert config.camera_index == 2
    assert config.selected_arm is Arm.LEFT


def test_main_is_a_non_interactive_smoke_check() -> None:
    assert main([]) == 0
