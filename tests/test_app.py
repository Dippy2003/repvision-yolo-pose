from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from repvision.app import build_parser, config_from_args, main, source_factory_from_args
from repvision.calibration import CalibrationProfile, CalibrationStore
from repvision.camera import Camera
from repvision.config import Arm
from repvision.pose_detector import PoseObservation, PoseStatus
from repvision.session import SessionLogger, SessionSummary
from repvision.video_source import VideoFileSource


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
            "--calibration-samples",
            "12",
            "--audio-cues",
        ]
    )

    config = config_from_args(args)

    assert config.model_name == "custom-pose.pt"
    assert config.camera_index == 2
    assert config.selected_arm is Arm.LEFT
    assert config.confidence_threshold == 0.65
    assert config.input_size == 480
    assert config.output_directory == Path("my-sessions")
    assert config.calibration_sample_target == 12
    assert config.audio_cues


def test_calibration_commands_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--calibrate", "--calibration-status"])


def test_cli_can_disable_saved_calibration_for_one_run() -> None:
    args = build_parser().parse_args(["--no-calibration"])

    assert args.no_calibration


def test_history_command_reports_saved_aggregate_sessions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    SessionLogger(tmp_path).save(
        SessionSummary(
            datetime(2026, 8, 17),
            "bicep_curl",
            Arm.RIGHT,
            60.0,
            10,
            1,
            2.0,
        )
    )

    assert main(["--history", "--output-directory", str(tmp_path)]) == 0

    output = capsys.readouterr().out
    assert "Sessions: 1" in output
    assert "Total reps: 10" in output


def test_history_command_rejects_invalid_recent_limit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--history", "--history-limit", "0"])

    assert exit_info.value.code == 2
    assert "history_limit must be positive" in capsys.readouterr().err


def test_calibration_status_reports_missing_selected_arm(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "calibration.json"

    assert main(["--calibration-status", "--calibration-file", str(path)]) == 0

    output = capsys.readouterr().out
    assert "No calibration saved for right arm" in output
    assert str(path) in output


def test_calibration_status_reports_saved_aggregate_profile(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "calibration.json"
    CalibrationStore(path).save(
        CalibrationProfile(
            Arm.LEFT,
            40.0,
            165.0,
            50.0,
            155.0,
            20,
            datetime(2026, 8, 17, tzinfo=UTC),
        )
    )

    assert main(
        [
            "--calibration-status",
            "--arm",
            "left",
            "--calibration-file",
            str(path),
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "arm=left" in output
    assert "curled=40.0" in output
    assert "extended=165.0" in output


def test_reset_calibration_removes_only_selected_arm(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "calibration.json"
    store = CalibrationStore(path)
    timestamp = datetime(2026, 8, 17, tzinfo=UTC)
    store.save(CalibrationProfile(Arm.RIGHT, 40, 165, 50, 155, 20, timestamp))
    store.save(CalibrationProfile(Arm.LEFT, 42, 164, 52, 154, 20, timestamp))

    assert main(
        ["--reset-calibration", "--arm", "left", "--calibration-file", str(path)]
    ) == 0

    assert store.load(Arm.LEFT) is None
    assert store.load(Arm.RIGHT) is not None
    assert "Calibration removed for left arm" in capsys.readouterr().out


def test_reset_calibration_reports_absent_profile(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(
        [
            "--reset-calibration",
            "--calibration-file",
            str(tmp_path / "missing.json"),
        ]
    ) == 0

    assert "Calibration not found for right arm" in capsys.readouterr().out


def test_main_runs_guided_calibration_and_reports_profile(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "calibration.json"
    saved = CalibrationProfile(
        Arm.RIGHT,
        40.0,
        165.0,
        50.0,
        155.0,
        20,
        datetime(2026, 8, 17, tzinfo=UTC),
    )

    with patch(
        "repvision.app.run_guided_calibration", return_value=saved
    ) as run:
        assert main(["--calibrate", "--calibration-file", str(path)]) == 0

    assert run.call_args.kwargs["store"].path == path
    assert "Calibration saved: arm=right" in capsys.readouterr().out


def test_main_starts_live_workout(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("repvision.app.run_workout", return_value="outputs/sessions.csv"):
        assert main(["--no-calibration"]) == 0

    assert "Aggregate session saved" in capsys.readouterr().out


def test_main_supplies_saved_profiles_to_workout(
    tmp_path: Path,
) -> None:
    path = tmp_path / "calibration.json"
    saved = CalibrationProfile(
        Arm.RIGHT,
        42.0,
        164.0,
        52.0,
        154.0,
        20,
        datetime(2026, 8, 17, tzinfo=UTC),
    )
    CalibrationStore(path).save(saved)

    with patch(
        "repvision.app.run_workout", return_value="outputs/sessions.csv"
    ) as run:
        assert main(["--calibration-file", str(path)]) == 0

    assert run.call_args.kwargs["calibration_profiles"] == {Arm.RIGHT: saved}


def test_main_calibration_bypass_supplies_no_profiles(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    saved = CalibrationProfile(
        Arm.RIGHT,
        42.0,
        164.0,
        52.0,
        154.0,
        20,
        datetime(2026, 8, 17, tzinfo=UTC),
    )
    CalibrationStore(path).save(saved)

    with patch(
        "repvision.app.run_workout", return_value="outputs/sessions.csv"
    ) as run:
        assert main(
            ["--calibration-file", str(path), "--no-calibration"]
        ) == 0

    assert run.call_args.kwargs["calibration_profiles"] == {}


def test_main_builds_local_video_source(capsys: pytest.CaptureFixture[str]) -> None:
    with patch(
        "repvision.app.run_workout", return_value="outputs/sessions.csv"
    ) as run:
        assert main(["--video", "curl.mp4", "--no-calibration"]) == 0

    source_factory = run.call_args.kwargs["source_factory"]
    source = source_factory()
    assert isinstance(source, VideoFileSource)
    assert source.path == Path("curl.mp4")
    assert "Aggregate session saved" in capsys.readouterr().out


@pytest.mark.parametrize(
    "command",
    [
        "--check-camera",
        "--check-pose",
        "--calibrate",
        "--calibration-status",
        "--reset-calibration",
    ],
)
def test_video_input_rejects_incompatible_command(
    command: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--video", "curl.mp4", command])

    assert exit_info.value.code == 2
    assert "cannot be combined" in capsys.readouterr().err


def test_source_factory_defaults_to_configured_camera() -> None:
    args = build_parser().parse_args(["--camera-index", "3"])
    config = config_from_args(args)

    source = source_factory_from_args(args, config)()

    assert isinstance(source, Camera)
    assert source.index == 3


def test_main_runs_headless_pipeline_benchmark(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = object()
    with (
        patch("repvision.app.run_benchmark", return_value=result) as run,
        patch("repvision.app.format_benchmark", return_value="benchmark report"),
    ):
        assert main(
            [
                "--benchmark",
                "--benchmark-frames",
                "12",
                "--warmup-frames",
                "3",
                "--no-calibration",
            ]
        ) == 0

    assert run.call_args.kwargs["measured_frames"] == 12
    assert run.call_args.kwargs["warmup_frames"] == 3
    assert capsys.readouterr().out.strip() == "benchmark report"


def test_main_reports_invalid_benchmark_frame_count(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--benchmark", "--benchmark-frames", "0"])

    assert exit_info.value.code == 2
    assert "measured_frames must be positive" in capsys.readouterr().err


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
        assert main(["--check-pose", "--arm", "left", "--no-calibration"]) == 0

    output = capsys.readouterr().out
    assert "people=0" in output
    assert "status=no_person" in output
    assert "arm=left" in output
    assert "angle=unavailable" in output
    assert "stage=unknown" in output
    assert "reps=0" in output
    assert "thresholds=default" in output


def test_pose_check_reports_personalized_thresholds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "calibration.json"
    CalibrationStore(path).save(
        CalibrationProfile(
            Arm.RIGHT,
            42.0,
            164.0,
            52.0,
            154.0,
            20,
            datetime(2026, 8, 17, tzinfo=UTC),
        )
    )
    observation = PoseObservation((), None, None, PoseStatus.NO_PERSON)

    with patch("repvision.app.check_pose", return_value=observation):
        assert main(
            ["--check-pose", "--calibration-file", str(path)]
        ) == 0

    assert "thresholds=personalized" in capsys.readouterr().out


def test_main_reports_invalid_configuration_as_cli_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--confidence", "1.5"])

    assert exit_info.value.code == 2
    assert "confidence_threshold must be between 0 and 1" in capsys.readouterr().err
