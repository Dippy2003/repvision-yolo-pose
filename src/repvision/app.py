"""Command-line entry point for RepVision."""

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

from repvision.benchmark import BenchmarkError, format_benchmark, run_benchmark
from repvision.calibration import (
    CalibrationError,
    CalibrationStorageError,
    CalibrationStore,
    calibrated_config_for_arm,
    format_calibration_profile,
)
from repvision.calibration_workflow import run_guided_calibration
from repvision.camera import Camera, CameraError
from repvision.config import AppConfig, Arm
from repvision.display import DisplayError
from repvision.frame_source import FrameSource, FrameSourceError
from repvision.history import format_session_history
from repvision.pose_detector import PoseDetector, PoseDetectorError, PoseObservation
from repvision.rep_counter import CurlTracker
from repvision.session import SessionLogError, SessionLogger
from repvision.video_source import VideoFileSource
from repvision.workout import run_workout


def build_parser() -> argparse.ArgumentParser:
    """Build the RepVision command-line parser."""
    defaults = AppConfig()
    parser = argparse.ArgumentParser(
        prog="repvision",
        description="Run local bicep-curl tracking with YOLO Pose.",
    )
    parser.add_argument("--model", default=defaults.model_name)
    parser.add_argument("--camera-index", type=int, default=defaults.camera_index)
    parser.add_argument(
        "--arm",
        type=Arm,
        choices=list(Arm),
        default=defaults.selected_arm,
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=defaults.confidence_threshold,
        help="minimum confidence for shoulder, elbow, and wrist",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=defaults.input_size,
        help="square model inference size in pixels",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=defaults.output_directory,
        help="directory for aggregate session CSV data",
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="analyze a local video instead of opening a webcam",
    )
    diagnostics = parser.add_mutually_exclusive_group()
    diagnostics.add_argument(
        "--check-camera",
        action="store_true",
        help="open the camera, read one frame, and exit",
    )
    diagnostics.add_argument(
        "--check-pose",
        action="store_true",
        help="load the model, analyze one camera frame, and exit",
    )
    diagnostics.add_argument(
        "--benchmark",
        action="store_true",
        help="measure the local pipeline without opening a display",
    )
    diagnostics.add_argument(
        "--calibrate",
        action="store_true",
        help="open the camera and create a selected-arm calibration",
    )
    diagnostics.add_argument(
        "--calibration-status",
        action="store_true",
        help="show the selected arm's saved calibration and exit",
    )
    diagnostics.add_argument(
        "--reset-calibration",
        action="store_true",
        help="remove the selected arm's saved calibration and exit",
    )
    diagnostics.add_argument(
        "--history",
        action="store_true",
        help="show aggregate workout history and exit",
    )
    parser.add_argument(
        "--benchmark-frames",
        type=int,
        default=30,
        help="number of measured frames for --benchmark",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=2,
        help="unmeasured warm-up frames for --benchmark",
    )
    parser.add_argument(
        "--calibration-file",
        type=Path,
        help="override the user-local calibration profile path",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=defaults.calibration_sample_target,
        help="reliable frames captured at each calibration endpoint",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="ignore saved calibration profiles for this run",
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=5,
        help="number of recent workouts shown by --history",
    )
    parser.add_argument(
        "--audio-cues",
        action="store_true",
        help="emit a local terminal bell for reps and form warnings",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> AppConfig:
    """Convert parsed command-line values into validated settings."""
    return AppConfig(
        model_name=args.model,
        camera_index=args.camera_index,
        selected_arm=args.arm,
        confidence_threshold=args.confidence,
        input_size=args.input_size,
        output_directory=args.output_directory,
        calibration_sample_target=args.calibration_samples,
        audio_cues=args.audio_cues,
    )


def check_camera(config: AppConfig) -> tuple[int, ...]:
    """Read one frame and release the camera immediately."""
    with Camera(config.camera_index) as camera:
        return camera.read().shape


def check_pose(config: AppConfig) -> PoseObservation:
    """Load the model and analyze exactly one locally captured frame."""
    detector = PoseDetector(config)
    with Camera(config.camera_index) as camera:
        return detector.detect(camera.read())


def source_factory_from_args(
    args: argparse.Namespace, config: AppConfig
) -> Callable[[], FrameSource]:
    """Build a deferred webcam or local-video source factory."""
    if args.video is not None:
        return lambda: VideoFileSource(args.video)
    return lambda: Camera(config.camera_index)


def main(argv: Sequence[str] | None = None) -> int:
    """Run a diagnostic or start the continuous local workout."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.video is not None and (
        args.check_camera
        or args.check_pose
        or args.calibrate
        or args.calibration_status
        or args.reset_calibration
        or args.history
    ):
        parser.error("--video cannot be combined with the selected command")
    try:
        config = config_from_args(args)
    except ValueError as error:
        parser.error(str(error))
    calibration_store = CalibrationStore(args.calibration_file)
    if args.history:
        if args.history_limit < 1:
            parser.error("history_limit must be positive")
        try:
            summaries = SessionLogger(config.output_directory).load()
        except SessionLogError as error:
            parser.error(str(error))
        print(format_session_history(summaries, recent_limit=args.history_limit))
        return 0
    if args.calibration_status:
        try:
            profile = calibration_store.load(config.selected_arm)
        except CalibrationStorageError as error:
            parser.error(str(error))
        if profile is None:
            print(
                f"No calibration saved for {config.selected_arm.value} arm "
                f"({calibration_store.path})."
            )
        else:
            print(
                "Calibration: "
                f"{format_calibration_profile(profile)} "
                f"(file={calibration_store.path})."
            )
        return 0
    if args.reset_calibration:
        try:
            removed = calibration_store.reset(config.selected_arm)
        except CalibrationStorageError as error:
            parser.error(str(error))
        result = "removed" if removed else "not found"
        print(
            f"Calibration {result} for {config.selected_arm.value} arm "
            f"({calibration_store.path})."
        )
        return 0
    if args.calibrate:
        try:
            profile = run_guided_calibration(config, store=calibration_store)
        except (
            CalibrationError,
            CameraError,
            DisplayError,
            PoseDetectorError,
        ) as error:
            parser.error(str(error))
        print(
            "Calibration saved: "
            f"{format_calibration_profile(profile)} "
            f"(file={calibration_store.path})."
        )
        return 0
    if args.check_camera:
        try:
            frame_shape = check_camera(config)
        except CameraError as error:
            parser.error(str(error))
        print(f"Camera check passed (frame shape={frame_shape}).")
        return 0
    try:
        calibration_profiles = (
            {} if args.no_calibration else calibration_store.load_all()
        )
    except CalibrationStorageError as error:
        parser.error(str(error))
    calibrated_config = calibrated_config_for_arm(
        config,
        config.selected_arm,
        calibration_profiles,
    )
    if args.check_pose:
        try:
            observation = check_pose(calibrated_config)
        except (CameraError, PoseDetectorError) as error:
            parser.error(str(error))
        curl_update = CurlTracker(calibrated_config).update(observation.selected_arm)
        angle_text = (
            "unavailable"
            if curl_update.smoothed_angle is None
            else f"{curl_update.smoothed_angle:.1f}"
        )
        threshold_source = (
            "personalized"
            if config.selected_arm in calibration_profiles
            else "default"
        )
        print(
            "Pose check passed "
            f"(people={len(observation.persons)}, status={observation.status.value}, "
            f"arm={calibrated_config.selected_arm.value}, angle={angle_text}, "
            f"stage={curl_update.stage.value}, reps={curl_update.count}, "
            f"thresholds={threshold_source})."
        )
        return 0
    if args.benchmark:
        try:
            result = run_benchmark(
                calibrated_config,
                source_factory_from_args(args, calibrated_config),
                measured_frames=args.benchmark_frames,
                warmup_frames=args.warmup_frames,
            )
        except (
            BenchmarkError,
            FrameSourceError,
            PoseDetectorError,
            ValueError,
        ) as error:
            parser.error(str(error))
        print(format_benchmark(result))
        return 0
    try:
        source_factory = source_factory_from_args(args, config)
        session_path = run_workout(
            config,
            source_factory=source_factory,
            calibration_profiles=calibration_profiles,
        )
    except (
        DisplayError,
        FrameSourceError,
        PoseDetectorError,
        SessionLogError,
    ) as error:
        parser.error(str(error))
    print(f"Workout complete. Aggregate session saved to {session_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
