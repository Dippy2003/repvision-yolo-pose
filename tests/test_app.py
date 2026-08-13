from pathlib import Path
from unittest.mock import patch

import pytest

from repvision.app import build_parser, config_from_args, main
from repvision.config import Arm
from repvision.pose_detector import PoseObservation, PoseStatus


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
            "--output-directory",
            "my-sessions",
        ]
    )

    config = config_from_args(args)

    assert config.model_name == "custom-pose.pt"
    assert config.camera_index == 2
    assert config.selected_arm is Arm.LEFT
    assert config.confidence_threshold == 0.65
    assert config.input_size == 480
    assert config.output_directory == Path("my-sessions")


def test_main_starts_live_workout(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("repvision.app.run_workout", return_value="outputs/sessions.csv"):
        assert main([]) == 0

    assert "Aggregate session saved" in capsys.readouterr().out


def test_main_builds_local_video_source(capsys: pytest.CaptureFixture[str]) -> None:
    with patch(
        "repvision.app.run_workout", return_value="outputs/sessions.csv"
    ) as run:
        assert main(["--video", "curl.mp4"]) == 0

    source_factory = run.call_args.kwargs["source_factory"]
    source = source_factory()
    assert isinstance(source, VideoFileSource)
    assert source.path == Path("curl.mp4")
    assert "Aggregate session saved" in capsys.readouterr().out


def test_camera_check_reports_single_frame_shape(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch("repvision.app.check_camera", return_value=(480, 640, 3)):
        assert main(["--check-camera"]) == 0

    assert "frame shape=(480, 640, 3)" in capsys.readouterr().out


def test_pose_check_reports_structured_result(
    capsys: pytest.CaptureFixture[str],
) -> None:
    observation = PoseObservation((), None, None, PoseStatus.NO_PERSON)
    with patch("repvision.app.check_pose", return_value=observation):
        assert main(["--check-pose", "--arm", "left"]) == 0

    output = capsys.readouterr().out
    assert "people=0" in output
    assert "status=no_person" in output
    assert "arm=left" in output
    assert "angle=unavailable" in output
    assert "stage=unknown" in output
    assert "reps=0" in output


def test_main_reports_invalid_configuration_as_cli_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--confidence", "1.5"])

    assert exit_info.value.code == 2
    assert "confidence_threshold must be between 0 and 1" in capsys.readouterr().err
