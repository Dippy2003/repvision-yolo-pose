from unittest.mock import patch

import pytest

from repvision.app import build_parser, config_from_args, main
from repvision.config import Arm


def test_cli_overrides_foundation_settings() -> None:
    args = build_parser().parse_args(
        [
            "--model",
            "custom-pose.pt",
            "--camera-index",
            "2",
            "--arm",
            "left",
            "--confidence",
            "0.65",
            "--input-size",
            "480",
        ]
    )

    config = config_from_args(args)

    assert config.model_name == "custom-pose.pt"
    assert config.camera_index == 2
    assert config.selected_arm is Arm.LEFT
    assert config.confidence_threshold == 0.65
    assert config.input_size == 480


def test_main_is_a_non_interactive_smoke_check() -> None:
    assert main([]) == 0


def test_camera_check_reports_single_frame_shape(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch("repvision.app.check_camera", return_value=(480, 640, 3)):
        assert main(["--check-camera"]) == 0

    assert "frame shape=(480, 640, 3)" in capsys.readouterr().out
